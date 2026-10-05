import numpy as np


def queue_cycles(active_jobs, satellite_count):
    """Only data already delivered to CPU contributes to its queue workload."""
    cycles = np.zeros(satellite_count)
    for job in active_jobs:
        if job.stage == "cpu":
            cycles[job.action.compute_sat] += job.remaining_cycles
    return cycles


def inflight_cycles(active_jobs, satellite_count):
    cycles = np.zeros(satellite_count)
    for job in active_jobs:
        if job.stage in ("tx", "prop"):
            cycles[job.action.compute_sat] += job.remaining_cycles
    return cycles


def workload_delay(cycles, capacities):
    """Q/F is a workload proxy, not an additional waiting time under sharing."""
    return np.asarray(cycles) / np.asarray(capacities)
