from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingAction:
    task_id: int
    compute_sat: int
    path: tuple

    def __post_init__(self):
        object.__setattr__(self, "path", tuple(self.path))
        if not self.path or len(self.path) != len(set(self.path)):
            raise ValueError("Routing paths must be nonempty and loop-free")
        if self.compute_sat != self.path[-1]:
            raise ValueError("Path must terminate at the computing satellite")

    @property
    def is_local(self):
        return len(self.path) == 1

    @property
    def hops(self):
        return len(self.path) - 1


@dataclass(frozen=True)
class CandidateAction:
    action: RoutingAction
    route_seconds: float
    workload_seconds: float
    execution_seconds: float
    deadline_margin_seconds: float
    bottleneck_bps: float
    topology_feasible: bool
    fully_checked: bool
    feasible: bool
    contact_margin_seconds: float = 0.0
    capacity_margin_ratio: float = 0.0
    search_truncated: bool = False

    @property
    def estimated_total_seconds(self):
        return self.route_seconds + self.workload_seconds + self.execution_seconds

    def features(self):
        # SI-valued features are normalized by state_builder, not silently here.
        return (self.action.hops, self.route_seconds, self.workload_seconds,
                self.execution_seconds, self.deadline_margin_seconds,
                self.bottleneck_bps, float(self.fully_checked), float(self.topology_feasible),
                self.contact_margin_seconds, self.capacity_margin_ratio)
