from ..routing.action_builder import RoutingAction


class LocalOnly:
    name, use_future = "local", False

    def select(self, observation):
        return {task.task_id: RoutingAction(task.task_id, task.source_sat, (task.source_sat,)) for task in observation.tasks}
