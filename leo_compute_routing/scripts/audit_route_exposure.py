"""Sample candidate contact exposure without policies, queues or future arrivals.

This is a preflight diagnostic, not a selected-route failure metric or an oracle.
Tasks are sampled evenly over the admission trace, without selecting contact epochs.
"""
import argparse
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config, fingerprint
from leo_routing.env.leo_env import generate_cpu_capacities
from leo_routing.evaluation.calibration import audit_scenario
from leo_routing.routing.candidate_builder import CandidateBuilder
from leo_routing.tasks.task_generator import generate_task_trace
from leo_routing.topology.topology_cache import generate_topology
from leo_routing.utils.io import save_csv, save_json, save_yaml


def inspect_candidates(config, topology, tasks, capacities, max_tasks):
    flat = [task for batch in tasks for task in batch]
    indices = np.linspace(0, len(flat) - 1, min(max_tasks, len(flat)), dtype=int) if flat else ()
    builder = CandidateBuilder(topology, capacities, config["routing"])
    empty_work = np.zeros(topology.satellite_count)
    rows = []
    risky_tasks = feasible_tasks = 0
    for index in indices:
        task = flat[index]
        candidates = builder.build(task, task.arrival_slot, empty_work, empty_work)
        feasible_tasks += int(any(candidate.feasible for candidate in candidates))
        risky_tasks += int(any(len(candidate.action.path) > 1 and not candidate.topology_feasible
                               for candidate in candidates))
        for candidate in candidates:
            if len(candidate.action.path) == 1:
                continue
            rows.append({"task_id": task.task_id, "arrival_slot": task.arrival_slot,
                         "destination": candidate.action.compute_sat, "path": list(candidate.action.path),
                         "hops": len(candidate.action.path) - 1,
                         "data_bits": task.data_bits, "total_cycles": task.total_cycles,
                         "deadline_s": task.deadline_seconds, "route_proxy_s": candidate.route_seconds,
                         "execution_proxy_s": candidate.execution_seconds,
                         "contact_loss_detected": not candidate.topology_feasible,
                         "fully_checked": candidate.fully_checked,
                         "empty_system_feasible": candidate.feasible})
    count, sampled = len(rows), len(indices)
    summary = {"sampled_tasks": sampled, "remote_candidate_count": count,
               "candidate_contact_loss_fraction": sum(r["contact_loss_detected"] for r in rows) / count if count else None,
               "candidate_unverified_fraction": sum(not r["fully_checked"] for r in rows) / count if count else None,
               "tasks_with_some_contact_risky_candidate_fraction": risky_tasks / sampled if sampled else None,
               "empty_system_some_candidate_feasible_fraction": feasible_tasks / sampled if sampled else None,
               "mean_remote_route_proxy_s": float(np.mean([r["route_proxy_s"] for r in rows])) if count else None,
               "note": "Evenly spaced tasks; all retained remote candidates; empty CPU/in-flight queues. "
                       "Reference-rate estimates, not realized service times or policy-selected route failure rates."}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[50, 51, 52])
    parser.add_argument("--max-tasks", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tasks < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Use positive max-tasks and distinct nonnegative seeds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is nonempty; use a new directory")
    config = load_config(args.config)
    if config["routing"]["lookahead_slots"] == 0:
        parser.error("Contact exposure requires a positive lookahead window")
    topology = generate_topology(config)
    args.output.mkdir(parents=True, exist_ok=True)
    save_yaml(args.output / "resolved_config.yaml", config)
    save_json(args.output / "study.json", {"config_sha256": fingerprint(config), "seeds": args.seeds,
              "max_sampled_tasks_per_seed": args.max_tasks, "uses_policy_ranking": False,
              "sampling": "evenly spaced indices over all admission tasks",
              "seed_scope": "task arrivals and CPU capacities; topology shared",
              "topology_signature": topology.signature})
    summaries = []
    for seed in args.seeds:
        current = load_config(args.config, ["simulation.seed=%s" % seed])
        tasks, capacities = generate_task_trace(current), generate_cpu_capacities(current)
        calibration = audit_scenario(current, topology, tasks, capacities)
        exposure, rows = inspect_candidates(current, topology, tasks, capacities, args.max_tasks)
        directory = args.output / ("seed_%s" % seed)
        save_json(directory / "calibration.json", calibration)
        save_json(directory / "route_exposure.json", exposure)
        save_csv(directory / "candidate_exposure.csv", rows)
        summary = {"seed": seed, "global_cpu_rho": calibration["expected_global_cpu_rho"],
                   "max_local_cpu_rho": max(calibration["expected_local_rho_per_satellite"]),
                   "optimistic_deadline_impossible_fraction": calibration["optimistic_intrinsic_deadline_impossible_fraction"],
                   "connected_slot_fraction": calibration["topology_diagnostics"]["connected_slot_fraction"],
                   "edge_state_changes": calibration["topology_diagnostics"]["edge_state_changes"],
                   "link_removal_window_fraction": calibration["current_link_removal_window_fraction"], **exposure}
        summaries.append(summary)
        save_csv(args.output / "summary.csv", summaries)
        print("seed=%s global_rho=%.3f max_local_rho=%.3f connected=%.3f candidate_contact_loss=%s" %
              (seed, summary["global_cpu_rho"], summary["max_local_cpu_rho"],
               summary["connected_slot_fraction"], summary["candidate_contact_loss_fraction"]), flush=True)
    print("Preflight diagnostics:", args.output.resolve())


if __name__ == "__main__":
    main()
