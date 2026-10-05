import numpy as np


def summarize(engine, slot_metrics):
    all_records = engine.task_records()
    window = engine.measurement_window
    records = [r for r in all_records if window is None or
               window[0] <= r["arrival_slot"] * engine.trace.slot_seconds < window[1]]
    measured_slots = [r for r in slot_metrics if window is None or
                      window[0] <= r["slot"] * engine.trace.slot_seconds < window[1]]
    count = len(records)
    completed = [row["completion_delay_s"] for row in records if row["status"] == "completed"]

    def fraction(predicate):
        return sum(predicate(row) for row in records) / count if count else 0.0

    def divide(first, second):
        return first / second if second else 0.0

    return {"task_count": count, "all_admitted_task_count": len(all_records), "completed_count": len(completed),
            "evaluation_start_s": window[0] if window else 0.0,
            "evaluation_end_s": window[1] if window else engine.now,
            "utilization_measurement_seconds": engine.measured_seconds,
            "mean_completion_delay_s": float(np.mean(completed)) if completed else None,
            "p95_completion_delay_s": float(np.percentile(completed, 95)) if completed else None,
            "mean_sojourn_s": float(np.mean([r["sojourn_s"] for r in records])) if records else None,
            "success_rate": fraction(lambda r: r["success"]),
            "completion_rate": fraction(lambda r: r["status"] == "completed"),
            "deadline_violation_rate": fraction(lambda r: r["deadline_missed"]),
            "route_failure_rate": fraction(lambda r: r["status"] == "route_failed"),
            "offloaded_route_failure_rate": divide(sum(r["status"] == "route_failed" for r in records),
                                                   sum(r["hops"] > 0 for r in records)),
            "censored_rate": fraction(lambda r: r["status"] == "censored"),
            "timed_out_rate": fraction(lambda r: r["status"] == "timed_out"),
            "mean_hops": float(np.mean([r["hops"] for r in records])) if records else 0.0,
            "cpu_utilization": divide(engine.measured_cpu_busy_integral, engine.measured_cpu_total_integral),
            "link_utilization": divide(engine.measured_link_busy_integral, engine.measured_link_total_integral),
            "episode_cpu_utilization": divide(engine.cpu_busy_capacity_integral, engine.cpu_total_capacity_integral),
            "episode_link_utilization": divide(engine.link_busy_capacity_integral, engine.link_total_capacity_integral),
            "mean_queue_cycles": float(np.mean([r["mean_queue_cycles"] for r in measured_slots])) if measured_slots else 0.0,
            "mean_queue_delay_variance": float(np.mean([r["queue_delay_variance"] for r in measured_slots])) if measured_slots else 0.0,
            "max_cpu_budget_ratio": engine.max_cpu_budget_ratio,
            "max_link_budget_ratio": engine.max_link_budget_ratio,
            "total_reward": float(sum(r["reward"] for r in slot_metrics)),
            "simulation_seconds": engine.now,
            "cpu_cycles_processed": engine.cpu_cycles_processed,
            "transmitted_bits": engine.transmitted_bits}
