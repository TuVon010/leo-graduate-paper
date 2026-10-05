"""Pre-run diagnostics, not automatic parameter tuning or literature validation."""
import math

import numpy as np


def audit_scenario(config, topology, tasks, cpu_capacities):
    sim, settings = config["simulation"], config["tasks"]
    count, dt = topology.satellite_count, sim["slot_seconds"]
    cpus = np.asarray(cpu_capacities)
    source_probability = np.full(count, (1 - settings["hotspot_probability"]) / count)
    hotspots = settings["hotspot_satellites"]
    if hotspots:
        source_probability[hotspots] += settings["hotspot_probability"] / len(hotspots)
    arrival_rate = settings["arrival_rate_per_slot"] / dt
    mean_bits = float(np.mean(settings["data_bits"]))
    # Independent uniform draws of D and cycles/bit in the task generator.
    mean_cycles = mean_bits * float(np.mean(settings["cycles_per_bit"]))
    rho = arrival_rate * mean_cycles * source_probability / cpus
    flat = [task for batch in tasks for task in batch]
    observed_work = np.zeros(count)
    for task in flat:
        observed_work[task.source_sat] += task.total_cycles
    duration = sim["slots"] * dt
    observed_rho = observed_work / duration / cpus
    optimistic_impossible = int(sum(task.total_cycles / cpus.max() > task.deadline_seconds for task in flat))
    local_intrinsic_impossible = int(sum(task.total_cycles / cpus[task.source_sat] > task.deadline_seconds
                                        for task in flat))
    adjacency = topology.adjacency[:sim["slots"]]
    propagation = topology.distances[:sim["slots"]][adjacency] / 299792458.0
    mean_propagation = float(propagation.mean()) if len(propagation) else None
    horizon = config["routing"]["lookahead_slots"]
    removal_windows = 0
    # Only count losses of links present at the decision snapshot, not additions.
    for slot in range(sim["slots"]):
        future = topology.adjacency[slot:min(topology.slots, slot + horizon + 1)]
        removal_windows += int(np.any(topology.adjacency[slot] & ~future))
    omega = math.sqrt(3.986004418e14 / (6371000.0 + config["topology"]["altitude_m"]) ** 3)
    network_proxy = (4 * (mean_bits / (config["topology"]["link_capacity_bps"] *
                                     config["routing"]["reference_rate_fraction"]) + mean_propagation)
                     if mean_propagation is not None else None)
    warnings = []
    if np.any(rho >= 1):
        warnings.append("Expected local-only rho >= 1: sustained local backlog is expected; label overload explicitly.")
    if optimistic_impossible:
        warnings.append("Some tasks miss deadlines even on the fastest sampled CPU without communication or contention.")
    if removal_windows == 0 and horizon:
        warnings.append("No current-link removal inside any configured prediction window; future routing gain is unestablished.")
    if config["topology"]["mode"] == "periodic":
        warnings.append("Synthetic contact topology: suitable for correctness/stress checks, not physical-orbit evidence.")
    warnings.append("CPU cycles/s, task distributions and workload intensity remain engineering assumptions pending calibration.")
    return {
        "schema_version": 1, "seed": sim["seed"], "task_count": len(flat),
        "admission_seconds": duration, "arrival_rate_per_second": arrival_rate,
        "expected_task_cycles": mean_cycles, "source_probability": source_probability.tolist(),
        "expected_hotspot_task_fraction": float(source_probability[hotspots].sum()),
        "expected_local_rho_per_satellite": rho.tolist(),
        "observed_local_workload_rho_per_satellite": observed_rho.tolist(),
        "expected_global_cpu_rho": float(arrival_rate * mean_cycles / cpus.sum()),
        "observed_global_cpu_rho": float(observed_work.sum() / duration / cpus.sum()),
        "optimistic_intrinsic_deadline_impossible_count": optimistic_impossible,
        "optimistic_intrinsic_deadline_impossible_fraction": optimistic_impossible / len(flat) if flat else 0.0,
        "source_local_intrinsic_deadline_impossible_count": local_intrinsic_impossible,
        "worst_configured_task_fastest_sampled_cpu_seconds":
            settings["data_bits"][1] * settings["cycles_per_bit"][1] / float(cpus.max()),
        "expected_task_exclusive_mean_cpu_seconds": mean_cycles / float(cpus.mean()),
        "reference_4hop_network_proxy_seconds": network_proxy,
        "reference_4hop_proxy_note": "4 equal mean-edge hops, reference fraction; not measured routes or shared-service latency",
        "lookahead_seconds": horizon * dt,
        "circular_orbit_degrees_in_lookahead": math.degrees(omega * horizon * dt),
        "current_link_removal_window_fraction": removal_windows / sim["slots"],
        "topology_diagnostics": topology.diagnostics(sim["slots"]),
        "deadline_lower_bound_note": "L/max(F) is an optimistic lower bound, not a complete reachability/route/deadline feasibility test",
        "warnings": warnings}
