from .allocator import allocate_capacity


def allocate_cpu(jobs, capacities, mode="sqrt"):
    groups = {}
    for job in jobs:
        if job.stage == "cpu":
            groups.setdefault(job.action.compute_sat, {})[job.task.task_id] = job.task.total_cycles
    return {satellite: allocate_capacity(workloads, float(capacities[satellite]), mode)
            for satellite, workloads in groups.items()}
