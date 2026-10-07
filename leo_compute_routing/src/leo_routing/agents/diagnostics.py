"""Separate requested satellite decisions, router fallbacks and real failures."""
def add_counts(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def decision_metrics(counts):
    decisions = counts.get("decision_count", 0)
    destinations = counts.get("destination_count", 0)
    return {**counts,
        "predicted_invalid_action_rate": counts.get("predicted_invalid_selection_count", 0) / decisions if decisions else 0.0,
        "fallback_rate": counts.get("fallback_count", 0) / decisions if decisions else 0.0,
        "destination_excluded_fraction": counts.get("masked_destination_count", 0) / destinations if destinations else 0.0,
        "blocked_probability_mass": counts.get("blocked_probability_mass_sum", 0) / decisions if decisions else 0.0,
        "route_search_truncated_rate": counts.get("route_search_truncated_count", 0) / decisions if decisions else 0.0,
        "router_fallback_rate": counts.get("router_fallback_count", 0) / decisions if decisions else 0.0}
