from ..network.link_model import edge_key
from .allocator import allocate_capacity


def allocate_links(jobs, graph, mode="sqrt"):
    groups = {}
    for job in jobs:
        if job.stage == "tx":
            first, second = job.action.path[job.hop:job.hop + 2]
            edge = edge_key(first, second)
            if not graph.has_edge(*edge):
                raise ValueError("Cannot allocate an unavailable link")
            groups.setdefault(edge, {})[job.task.task_id] = job.task.data_bits
    return {edge: allocate_capacity(workloads, graph.edges[edge]["capacity_bps"], mode)
            for edge, workloads in groups.items()}
