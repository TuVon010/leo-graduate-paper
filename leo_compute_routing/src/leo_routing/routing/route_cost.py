def computing_aware_cost(candidate, route_weight=1.0, queue_weight=1.0, cpu_weight=1.0):
    return (route_weight * candidate.route_seconds + queue_weight * candidate.workload_seconds +
            cpu_weight * candidate.execution_seconds)
