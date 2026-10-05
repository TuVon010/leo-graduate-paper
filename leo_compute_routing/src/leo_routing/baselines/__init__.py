from .local_only import LocalOnly
from .shortest_path import ShortestOffload
from .least_load import LeastLoad
from .computing_aware import ComputingAware
from .batch_greedy import BatchGreedy

POLICY_NAMES = ("local", "shortest_offload", "least_load", "computing_aware", "computing_aware_future",
                "batch_greedy", "batch_greedy_future")


def make_policy(name):
    if name == "local":
        return LocalOnly()
    if name == "shortest_offload":
        return ShortestOffload()
    if name == "least_load":
        return LeastLoad()
    if name == "computing_aware":
        return ComputingAware()
    if name == "computing_aware_future":
        return ComputingAware(use_future=True)
    if name == "batch_greedy":
        return BatchGreedy()
    if name == "batch_greedy_future":
        return BatchGreedy(use_future=True)
    raise ValueError("Unknown baseline: %s" % name)
