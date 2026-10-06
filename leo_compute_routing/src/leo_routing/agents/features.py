"""Copy ragged policy inputs at action time; no Torch dependency or future arrivals.

Immutable snapshots preserve each autoregressive prefix and mask during PPO
updates. Scaling is fixed in the checkpoint, not fitted on evaluation results.
"""
from dataclasses import dataclass

import numpy as np

from ..network.link_model import edge_key
from ..routing.reservations import ReservationCalendar

FEATURE_SCHEMA = 2
NODE_DIM, EDGE_DIM, CONTEXT_DIM, TASK_DIM, CANDIDATE_DIM = 8, 4, 12, 5, 18
CONTACT_COMPLETION_INDEX = 13  # signed log(1 + calendar completion seconds / time_scale)


def frozen_array(values, dtype=np.float32):
    array = np.array(values, dtype=dtype, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError("Nonfinite policy features")
    array.setflags(write=False)
    return array


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass(frozen=True)
class GraphInput:
    nodes: np.ndarray
    edge_index: np.ndarray
    edges: np.ndarray
    context: np.ndarray


@dataclass(frozen=True)
class DecisionInput:
    task_id: int
    source: int
    destinations: np.ndarray
    path_pool: np.ndarray
    task: np.ndarray
    candidates: np.ndarray
    mask: np.ndarray
    fallback: bool
    predicted_feasible: np.ndarray


@dataclass(frozen=True)
class BatchInput:
    graph: GraphInput
    decisions: tuple
    actions: tuple

    @property
    def has_choice(self):
        return any(np.count_nonzero(d.mask) > 1 for d in self.decisions)


class FeatureBuilder:
    def __init__(self, config, settings):
        self.config, self.settings = config, settings
        self.time_scale = settings["time_scale_seconds"]
        training_nodes = config["topology"]["planes"] * config["topology"]["sats_per_plane"]
        self.per_node_task_scale = max(1.0, config["tasks"]["arrival_rate_per_slot"] / training_nodes)
        self.max_bits = config["tasks"]["data_bits"][1]
        self.max_complexity = config["tasks"]["cycles_per_bit"][1]

    def graph_input(self, observation):
        pending_cycles = np.zeros(len(observation.cpu_capacities))
        pending_count = np.zeros_like(pending_cycles)
        for task in observation.tasks:
            pending_cycles[task.source_sat] += task.total_cycles
            pending_count[task.source_sat] += 1
        batch_scale = self.per_node_task_scale * len(observation.cpu_capacities)
        nodes = np.column_stack((observation.node_features, pending_cycles / observation.cpu_capacities / self.time_scale,
                                 pending_count / self.per_node_task_scale))
        nodes[:, [0, 2]] /= self.time_scale
        nodes[:, [0, 2, 6, 7]] = signed_log(nodes[:, [0, 2, 6, 7]])
        edges = np.array(observation.edge_features, copy=True)
        edges[:, 2] = signed_log(edges[:, 2] / self.time_scale)
        jobs, tasks = observation.active_jobs, observation.tasks
        sim = self.config["simulation"]
        context = [max(0.0, 1 - observation.time_seconds / (sim["slots"] * sim["slot_seconds"])),
                   len(tasks) / batch_scale,
                   sum(j.stage == "cpu" for j in jobs) / batch_scale,
                   sum(j.stage == "tx" for j in jobs) / batch_scale,
                   sum(j.stage == "prop" for j in jobs) / batch_scale,
                   sum(j.deadline_reported for j in jobs) / batch_scale,
                   sum(j.remaining_cycles for j in jobs) / observation.cpu_capacities.sum() / self.time_scale,
                   sum(j.remaining_bits for j in jobs if j.stage == "tx") / self.max_bits / batch_scale,
                   sum(t.data_bits for t in tasks) / self.max_bits / batch_scale,
                   sum(t.total_cycles for t in tasks) / (self.max_bits * self.max_complexity * batch_scale),
                   min((t.deadline_seconds for t in tasks), default=0) / self.time_scale,
                   min((j.deadline_remaining_seconds for j in jobs), default=0) / self.time_scale]
        return GraphInput(frozen_array(nodes), frozen_array(observation.edge_index, np.int64),
                          frozen_array(edges), frozen_array(signed_log(np.asarray(context))))

    def decision_input(self, observation, task, position, reserved_cpu, reserved_links, calendar=None):
        items = observation.candidates[task.task_id]
        values = np.array(observation.candidate_features[task.task_id], copy=True)
        values[:, 1:5] /= self.time_scale
        values[:, 8] /= self.time_scale
        extras, pools, destinations, estimates = [], [], [], []
        calendar = calendar or ReservationCalendar.from_observation(observation)
        base_link_work = {}
        for job in observation.active_jobs:
            if job.stage == "tx":
                edge = edge_key(*job.path[job.hop:job.hop + 2])
                base_link_work[edge] = base_link_work.get(edge, 0.0) + job.remaining_bits
        costs = []
        for item in items:
            action = item.action
            cpu_extra = reserved_cpu[action.compute_sat] / observation.cpu_capacities[action.compute_sat]
            path_edges = [edge_key(i, j) for i, j in zip(action.path, action.path[1:])]
            link_extra = sum(reserved_links.get(e, 0.0) / observation.graph.edges[e]["capacity_bps"] for e in path_edges)
            current_links = sum(base_link_work.get(e, 0.0) / observation.graph.edges[e]["capacity_bps"] for e in path_edges)
            estimate = calendar.estimate(task, action)
            estimates.append(estimate)
            extras.append((cpu_extra / self.time_scale, link_extra / self.time_scale, current_links / self.time_scale,
                           estimate.completion_seconds / self.time_scale,
                           estimate.deadline_margin_seconds / self.time_scale,
                           estimate.route.contact_margin_seconds / self.time_scale,
                           estimate.route.capacity_margin_ratio, float(estimate.fully_checked)))
            costs.append(item.estimated_total_seconds + cpu_extra + link_extra + current_links)
            pool = np.zeros(len(observation.cpu_capacities))
            pool[list(action.path)] = 1 / len(action.path)
            pools.append(pool)
            destinations.append(action.compute_sat)
        values = np.column_stack((values, extras))
        indices = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        values[:, indices] = signed_log(values[:, indices])
        predicted_feasible = np.asarray([estimate.contact_ok and estimate.deadline_margin_seconds >= -1e-10
                                        for estimate in estimates], dtype=bool)
        mask = np.ones(len(items), dtype=bool)
        fallback = False
        if self.settings["use_mask"] and self.settings["shield_mode"] != "none":
            if self.settings["shield_mode"] == "contact":
                mask = np.asarray([estimate.feasible for estimate in estimates], dtype=bool)
                fallback = not mask.any()
            else:
                mask = np.array(observation.feasibility_masks[task.task_id], dtype=bool)
                if observation.deadline_mask_enabled:
                    mask &= np.asarray(costs) <= task.deadline_seconds
                fallback = task.task_id in observation.fallback_task_ids or not mask.any()
            if not mask.any():
                mask[0] = True
        task_values = [task.data_bits / self.max_bits, task.cycles_per_bit / self.max_complexity,
                       task.deadline_seconds / self.time_scale,
                       task.total_cycles / observation.cpu_capacities.mean() / self.time_scale,
                       position / max(1, len(observation.tasks))]
        return DecisionInput(task.task_id, task.source_sat, frozen_array(destinations, np.int64),
                             frozen_array(pools), frozen_array(signed_log(np.asarray(task_values))),
                             frozen_array(values), frozen_array(mask, bool), fallback,
                             frozen_array(predicted_feasible, bool))

    def book(self, action, task, reserved_cpu, reserved_links, calendar=None):
        if self.settings["use_reservations"]:
            reserved_cpu[action.compute_sat] += task.total_cycles
            for i, j in zip(action.path, action.path[1:]):
                edge = edge_key(i, j)
                reserved_links[edge] = reserved_links.get(edge, 0.0) + task.data_bits
            if calendar is not None:
                calendar.commit(task, action)
