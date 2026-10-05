"""Sequential virtual CPU/link workload bookings inside an atomic batch.

Workload/capacity is a scoring proxy, not an extra delay charged by the engine.
Booking downstream links conservatively counts commitments before they arrive.
"""
import numpy as np

from ..network.link_model import edge_key


class BatchGreedy:
    def __init__(self, use_future=False):
        self.use_future = use_future
        self.name = "batch_greedy_future" if use_future else "batch_greedy"

    def select(self, observation):
        reserved_cpu = np.zeros_like(observation.cpu_capacities)
        link_work = {}
        for job in observation.active_jobs:
            if job.stage == "tx":
                edge = edge_key(*job.path[job.hop:job.hop + 2])
                link_work[edge] = link_work.get(edge, 0.0) + job.remaining_bits
        actions = {}
        # Earliest absolute deadline first; tie-breaks are fully deterministic.
        for task in sorted(observation.tasks, key=lambda t: (t.deadline_seconds, t.task_id)):
            items = observation.candidates[task.task_id]
            costs = []
            for item in items:
                action = item.action
                link_penalty = sum(link_work.get(edge_key(i, j), 0.0) /
                                   observation.graph.edges[i, j]["capacity_bps"]
                                   for i, j in zip(action.path, action.path[1:]))
                costs.append(item.estimated_total_seconds + link_penalty +
                             reserved_cpu[action.compute_sat] / observation.cpu_capacities[action.compute_sat])
            indices = list(range(len(items)))
            if self.use_future:
                indices = [i for i in indices if observation.feasibility_masks[task.task_id][i]
                           and (not observation.deadline_mask_enabled or costs[i] <= task.deadline_seconds)]
                # Keep fallback explicit rather than claiming a feasible assignment.
                if not indices:
                    indices = [0]
            choice = min(indices, key=lambda i: (costs[i], i))
            actions[task.task_id] = choice
            action = items[choice].action
            reserved_cpu[action.compute_sat] += task.total_cycles
            for i, j in zip(action.path, action.path[1:]):
                edge = edge_key(i, j)
                link_work[edge] = link_work.get(edge, 0.0) + task.data_bits
        return actions
