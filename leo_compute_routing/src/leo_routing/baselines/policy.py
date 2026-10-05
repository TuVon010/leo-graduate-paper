from typing import Protocol


class RoutingPolicy(Protocol):
    name: str
    use_future: bool

    def select(self, observation) -> dict:
        """Return {task_id: candidate_index}; an empty batch returns {}."""
        ...
