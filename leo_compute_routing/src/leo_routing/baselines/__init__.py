from .local_only import LocalOnly
from .node_heuristics import NodeHeuristic

POLICY_NAMES = ("local", "shortest_offload", "least_load", "computing_aware", "computing_aware_future",
                "batch_greedy", "node_greedy")


def make_policy(name, config=None):
    if name == "local":
        return LocalOnly()
    if name not in POLICY_NAMES:
        raise ValueError("Unknown baseline: " + name)
    criterion = "network" if name == "shortest_offload" else "load" if name == "least_load" else "completion"
    future = name.endswith("_future") or name == "node_greedy"
    booking = name.startswith("batch_") or name == "node_greedy"
    return NodeHeuristic(name, criterion, future, booking, config)
