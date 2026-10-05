"""Counters distinguish prediction violations, masking, and actual outcomes."""


def add_counts(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def decision_metrics(counts):
    decisions, candidates = counts.get("decision_count", 0), counts.get("candidate_count", 0)
    return {**counts,
            "predicted_invalid_action_rate": counts.get("predicted_invalid_selection_count", 0) / decisions if decisions else 0.0,
            "fallback_rate": counts.get("fallback_count", 0) / decisions if decisions else 0.0,
            "shield_excluded_fraction": counts.get("masked_candidate_count", 0) / candidates if candidates else 0.0,
            "shield_blocked_probability_mass": counts.get("blocked_probability_mass_sum", 0) / decisions if decisions else 0.0,
            "candidate_search_truncated_rate": counts.get("search_truncated_task_count", 0) / decisions if decisions else 0.0}
