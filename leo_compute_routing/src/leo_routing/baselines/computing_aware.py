from ..routing.route_cost import computing_aware_cost


class ComputingAware:
    def __init__(self, use_future=False):
        self.use_future = use_future
        self.name = "computing_aware_future" if use_future else "computing_aware"

    def select(self, observation):
        actions = {}
        for task in observation.tasks:
            items = observation.candidates[task.task_id]
            indices = (list(i for i, permitted in enumerate(observation.feasibility_masks[task.task_id]) if permitted)
                       if self.use_future else list(range(len(items))))
            actions[task.task_id] = min(indices, key=lambda i: (computing_aware_cost(items[i]), i))
        return actions
