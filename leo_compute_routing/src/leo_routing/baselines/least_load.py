class LeastLoad:
    name, use_future = "least_load", False

    def select(self, observation):
        actions = {}
        for task in observation.tasks:
            items = observation.candidates[task.task_id]
            actions[task.task_id] = min(range(len(items)), key=lambda i: (items[i].workload_seconds,
                                                                       items[i].route_seconds, i))
        return actions
