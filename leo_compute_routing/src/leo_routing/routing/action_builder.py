from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingAction:
    task_id: int
    compute_sat: int
    path: tuple
    requested_compute_sat: object = None
    routing_rejection_reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, "path", tuple(self.path))
        if not self.path or len(self.path) != len(set(self.path)):
            raise ValueError("Routing paths must be nonempty and loop-free")
        if self.compute_sat != self.path[-1]:
            raise ValueError("Path must terminate at the computing satellite")
        if any(not isinstance(node, int) or isinstance(node, bool) or node < 0 for node in self.path):
            raise ValueError("Path nodes must be nonnegative integer satellite IDs")

    @property
    def is_local(self):
        return len(self.path) == 1

    @property
    def hops(self):
        return len(self.path) - 1
