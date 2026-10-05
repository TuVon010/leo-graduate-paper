from copy import deepcopy
from time import perf_counter

import numpy as np

from ..config import validate_config
from ..routing.action_builder import RoutingAction
from ..routing.candidate_builder import CandidateBuilder
from ..tasks.task_generator import generate_task_trace
from ..topology.topology_cache import generate_topology
from .event_engine import EventEngine
from .metrics import summarize
from .reward import compute_reward
from .state_builder import build_observation


def generate_cpu_capacities(config):
    rng = np.random.default_rng(np.random.SeedSequence([config["simulation"]["seed"], 2001]))
    count = config["topology"]["planes"] * config["topology"]["sats_per_plane"]
    return rng.uniform(*config["compute"]["cpu_cycles_per_second"], count)


class LeoEnv:
    """Structured reset/step API for variable-size BATCH actions.

    This class deliberately does not claim Gymnasium compliance: observations
    carry ragged graphs and candidate sets. A later RL adapter can define the
    matching spaces while all policies retain this common simulator.
    """
    def __init__(self, config, topology=None, task_trace=None, cpu_capacities=None):
        validate_config(config)
        self.config = deepcopy(config)
        self.topology = topology if topology is not None else generate_topology(config)
        self.task_trace = task_trace if task_trace is not None else generate_task_trace(config)
        self.cpu_capacities = (generate_cpu_capacities(config) if cpu_capacities is None
                               else np.asarray(cpu_capacities, dtype=float).copy())
        sim, topo = config["simulation"], config["topology"]
        if self.topology.satellite_count != topo["planes"] * topo["sats_per_plane"]:
            raise ValueError("Topology satellite count does not match configuration")
        if self.topology.slot_seconds != sim["slot_seconds"]:
            raise ValueError("Topology slot duration does not match configuration")
        if self.topology.slots < sim["slots"] + sim["drain_slots"] + 1:
            raise ValueError("Topology trace does not cover admission and drain phases")
        if len(self.task_trace) != sim["slots"]:
            raise ValueError("Task trace length does not match admission slots")
        ids = []
        for slot, batch in enumerate(self.task_trace):
            for task in batch:
                if task.arrival_slot != slot or task.source_sat >= self.topology.satellite_count:
                    raise ValueError("Invalid task trace slot or source")
                ids.append(task.task_id)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate task IDs in trace")
        self.engine = None
        self.observation = None

    def reset(self, *, seed=None):
        if seed is not None and seed != self.config["simulation"]["seed"]:
            raise ValueError("Use a new configured experiment to change a replay trace seed")
        sim = self.config["simulation"]
        warmup = self.config.get("evaluation", {}).get("warmup_slots", 0)
        measurement_window = (warmup * sim["slot_seconds"], sim["slots"] * sim["slot_seconds"])
        self.engine = EventEngine(self.topology, self.cpu_capacities,
                                  self.config["resource"]["allocation"],
                                  self.config["simulation"]["drop_at_deadline"], measurement_window)
        self.builder = CandidateBuilder(self.topology, self.cpu_capacities, self.config["routing"])
        self.slot, self.done, self.slot_metrics = 0, False, []
        self.observation_build_seconds = 0.0
        self.observation = self._observe()
        return self.observation, {"topology_signature": self.topology.signature}

    def _observe(self):
        started = perf_counter()
        tasks = self.task_trace[self.slot] if self.slot < len(self.task_trace) else ()
        observation = build_observation(self.engine, tasks, self.slot, self.builder, self.config)
        self.observation_build_seconds += perf_counter() - started
        return observation

    def step(self, actions):
        if self.engine is None or self.done:
            raise RuntimeError("Call reset before stepping an active episode")
        if not isinstance(actions, dict):
            raise TypeError("Batch actions must map task_id to candidate index or RoutingAction")
        tasks, normalized = self.observation.tasks, {}
        if set(actions) != {task.task_id for task in tasks}:
            raise ValueError("Action keys must exactly match the current task batch")
        for task in tasks:
            choice = actions[task.task_id]
            items = self.observation.candidates[task.task_id]
            if isinstance(choice, RoutingAction):
                if not any(candidate.action == choice for candidate in items):
                    raise ValueError("Action is not a current candidate")
                normalized[task.task_id] = choice
            elif isinstance(choice, (int, np.integer)) and not isinstance(choice, bool) and 0 <= choice < len(items):
                normalized[task.task_id] = items[int(choice)].action
            else:
                raise ValueError("Invalid candidate index/action")
        # Mask use is a POLICY choice, enabling controlled look-ahead ablations.
        self.engine.admit(tasks, normalized)
        interval = self.engine.advance((self.slot + 1) * self.config["simulation"]["slot_seconds"])
        reward = compute_reward(interval, self.config["reward"])
        queues = self.engine.queue_cycles()
        metric = {"slot": self.slot, "time_seconds": self.engine.now, "arrivals": len(tasks),
                  "active_tasks": len(self.engine.active), "reward": reward,
                  "mean_queue_cycles": float(queues.mean()),
                  "queue_delay_variance": float(np.var(queues / self.cpu_capacities)), **interval}
        self.slot_metrics.append(metric)
        self.slot += 1
        sim = self.config["simulation"]
        terminated = self.slot >= sim["slots"] and not self.engine.active
        truncated = not terminated and self.slot >= sim["slots"] + sim["drain_slots"]
        if truncated:
            self.engine.censor_remaining()
        self.done = terminated or truncated
        self.observation = None if self.done else self._observe()
        info = {"slot_metrics": metric}
        if self.done:
            info["episode_metrics"] = summarize(self.engine, self.slot_metrics)
        return self.observation, reward, terminated, truncated, info
