from dataclasses import dataclass
from types import MappingProxyType

import networkx as nx
import numpy as np

from ..routing.candidate_builder import effective_mask


def readonly(values, dtype=float):
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class JobSnapshot:
    task_id: int
    source_sat: int
    compute_sat: int
    path: tuple
    hop: int
    stage: str
    remaining_bits: float
    remaining_cycles: float
    seconds_to_wake: float
    deadline_remaining_seconds: float
    original_bits: float
    original_cycles: float
    deadline_reported: bool


@dataclass(frozen=True)
class Observation:
    slot: int
    time_seconds: float
    graph: object
    tasks: tuple
    candidates: object
    feasibility_masks: object
    fallback_task_ids: tuple
    cpu_capacities: np.ndarray
    cpu_queue_cycles: np.ndarray
    inflight_cycles: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    task_features: np.ndarray
    candidate_features: object
    active_jobs: tuple
    deadline_mask_enabled: bool
    contact_plan: object
    reference_rate_fraction: float
    allow_unverified_future: bool


def build_observation(engine, tasks, slot, candidate_builder, config):
    trace, capacities = engine.trace, engine.cpu_capacities
    queues, transit = engine.queue_cycles(), engine.inflight_cycles()
    candidates = {task.task_id: candidate_builder.build(task, slot, queues, transit) for task in tasks}
    masks, fallback = {}, []
    for task_id, items in candidates.items():
        masks[task_id], needed = effective_mask(items)
        if needed:
            fallback.append(task_id)
    positions = trace.positions[slot] / (6371000.0 + config["topology"]["altitude_m"])
    nodes = np.column_stack((queues / capacities, capacities / capacities.mean(), transit / capacities, positions))
    edge_index, edges = [], []
    graph = trace.graph(slot)
    horizon = config["routing"]["lookahead_slots"]
    link_work = {}
    for job in engine.active:
        if job.stage == "tx":
            link_work[job.edge] = link_work.get(job.edge, 0.0) + job.remaining_bits
    for i, j, attributes in graph.edges(data=True):
        end = min(trace.slots, slot + horizon + 1)
        availability = float(trace.adjacency[slot:end, i, j].mean())
        features = (attributes["distance_m"] / config["topology"]["max_isl_distance_m"],
                    attributes["capacity_bps"] / config["topology"]["link_capacity_bps"],
                    link_work.get((min(i, j), max(i, j)), 0.0) / attributes["capacity_bps"], availability)
        edge_index.extend(((i, j), (j, i)))
        edges.extend((features, features))
    task_features = [(task.data_bits / config["tasks"]["data_bits"][1],
                      task.cycles_per_bit / config["tasks"]["cycles_per_bit"][1],
                      task.deadline_seconds, task.total_cycles / capacities.mean()) for task in tasks]
    candidate_features = {}
    for task_id, items in candidates.items():
        features = np.asarray([candidate.features() for candidate in items])
        features[:, 0] /= max(1, config["routing"]["max_path_hops"])
        features[:, 5] /= config["topology"]["link_capacity_bps"]
        candidate_features[task_id] = readonly(features)
    active_jobs = tuple(JobSnapshot(
        job.task.task_id, job.task.source_sat, job.action.compute_sat, job.action.path, job.hop,
        job.stage, job.remaining_bits, job.remaining_cycles,
        max(0.0, job.wake_time - engine.now) if job.stage == "prop" else 0.0,
        job.absolute_deadline - engine.now, job.task.data_bits, job.task.total_cycles,
        job.deadline_reported) for job in engine.active)
    return Observation(slot, engine.now, nx.freeze(graph.copy()), tuple(tasks),
                       MappingProxyType(candidates), MappingProxyType(masks), tuple(fallback),
                       readonly(capacities), readonly(queues), readonly(transit), readonly(nodes),
                       readonly(np.asarray(edge_index, dtype=int).reshape(-1, 2).T, int),
                       readonly(np.asarray(edges).reshape(-1, 4)),
                       readonly(np.asarray(task_features).reshape(-1, 4)),
                       MappingProxyType(candidate_features), active_jobs, config["routing"]["deadline_mask"],
                       candidate_builder.contact_plan(slot), config["routing"]["reference_rate_fraction"],
                       config["routing"]["allow_unverified_future"])
