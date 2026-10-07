from typing import Protocol


class RoutingPolicy(Protocol):
    name: str
    use_future: bool

    def select(self, observation) -> dict:
        """Return {task_id: RoutingAction}; an empty batch returns {}."""
        ...
