class LocalOnly:
    name, use_future = "local", False

    def select(self, observation):
        return {task.task_id: 0 for task in observation.tasks}
