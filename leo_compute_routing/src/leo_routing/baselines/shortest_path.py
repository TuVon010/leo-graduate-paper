class ShortestOffload:
    """Network-only offloading: local is a fallback, not a zero-cost competitor.

    Including local in a pure communication minimization makes this identical
    to LocalOnly. The explicit name prevents misrepresenting that comparison.
    """
    name, use_future = "shortest_offload", False

    def select(self, observation):
        actions = {}
        for task in observation.tasks:
            items = observation.candidates[task.task_id]
            indices = [i for i, item in enumerate(items) if not item.action.is_local]
            actions[task.task_id] = (min(indices, key=lambda i: (items[i].route_seconds, i)) if indices else 0)
        return actions
