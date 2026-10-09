"""Copy ragged policy inputs at action time; no Torch dependency or future arrivals.

Immutable snapshots preserve each autoregressive prefix and mask during PPO
updates. Scaling is fixed in the checkpoint, not fitted on evaluation results.
"""
from dataclasses import dataclass

import numpy as np

import networkx as nx

FEATURE_SCHEMA = 3
NODE_DIM, EDGE_DIM, CONTEXT_DIM, TASK_DIM, DESTINATION_DIM = 8, 4, 12, 5, 7


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
    task: np.ndarray
    destination_features: np.ndarray
    mask: np.ndarray
    fallback: bool


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

    def decision_input(self, observation, task, position, reserved_cpu, distances=None):
        capacities = observation.cpu_capacities
        count = len(capacities)
        if distances is None:
            distances = nx.single_source_shortest_path_length(observation.graph, task.source_sat,
                                                             cutoff=observation.max_compute_hops)
        hops = np.array([distances.get(s, observation.max_compute_hops + 1) for s in range(count)])
        reachable = np.array([s in distances for s in range(count)])
        exclusive = task.total_cycles / capacities
        source_position = observation.node_features[task.source_sat, 3:6]
        spatial = np.linalg.norm(observation.node_features[:, 3:6] - source_position, axis=1)
        # Workload features are congestion indicators, not added actual FIFO waits.
        features = np.column_stack((observation.cpu_queue_cycles / capacities / self.time_scale,
            observation.inflight_cycles / capacities / self.time_scale,
            reserved_cpu / capacities / self.time_scale, exclusive / self.time_scale,
            spatial, hops / max(1, observation.max_compute_hops), reachable))
        features[:, :4] = signed_log(features[:, :4])
        mask = reachable.copy()
        if self.settings["use_mask"] and observation.deadline_mask_enabled:
            mask &= exclusive <= task.deadline_seconds
        fallback = not mask.any()
        if fallback:
            mask[task.source_sat] = True
        task_values = [task.data_bits / self.max_bits, task.cycles_per_bit / self.max_complexity,
                       task.deadline_seconds / self.time_scale,
                       task.total_cycles / capacities.mean() / self.time_scale,
                       position / max(1, len(observation.tasks))]
        return DecisionInput(task.task_id, task.source_sat, frozen_array(signed_log(np.asarray(task_values))),
                             frozen_array(features), frozen_array(mask, bool), fallback)
