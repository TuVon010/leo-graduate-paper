def compute_reward(interval, settings):
    """Latency holding cost supplies credit while tasks are still unfinished.

    Integral active task count equals aggregate task sojourn over an episode,
    including failed tasks until their terminal event. Normalize by a FIXED
    constant, not the policy-dependent number of completions in this slot.
    """
    return -(interval["holding_cost_seconds"] +
             settings["deadline_penalty"] * interval["new_deadline_misses"] +
             settings["route_failure_penalty"] * (interval["new_route_failures"] +
                                                  interval.get("new_routing_rejections", 0))) / settings["normalizer"]
