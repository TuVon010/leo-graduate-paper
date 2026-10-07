"""Sample selected-destination route exposure without policies, queues or future arrivals.

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
from leo_routing.env.event_engine import EventEngine
from leo_routing.env.state_builder import build_observation
from leo_routing.routing.reservations import ReservationCalendar
from leo_routing.routing.contact_aware_router import ContactAwareRouter
import networkx as nx
from leo_routing.tasks.task_generator import generate_task_trace
from leo_routing.topology.topology_cache import generate_topology
from leo_routing.utils.io import save_csv, save_json, save_yaml


def inspect_routes(config, topology, tasks, capacities, max_tasks):
    flat = [task for batch in tasks for task in batch]
    indices = np.linspace(0, len(flat) - 1, min(max_tasks, len(flat)), dtype=int) if flat else ()
    settings = dict(config["routing"])
    snapshot = ContactAwareRouter(dict(settings, mode="snapshot"))
    contact = ContactAwareRouter(dict(settings, mode="contact"))
    rows = []
    for index in indices:
        task = flat[index]
        # Empty diagnostic state at this arrival epoch: no historical workload.
        engine = EventEngine(topology, capacities)
        engine.now = task.arrival_slot * topology.slot_seconds
        obs = build_observation(engine, (task,), task.arrival_slot, config)
        calendar = ReservationCalendar.from_observation(obs)
        destinations = nx.single_source_shortest_path_length(obs.graph, task.source_sat,
                                                            cutoff=settings["max_compute_hops"])
        for destination in sorted(destinations):
            if destination == task.source_sat:
                continue
            old = snapshot.find_route(obs, task, destination, calendar)
            new = contact.find_route(obs, task, destination, calendar)
            prediction = old.prediction
            rows.append({"task_id": task.task_id, "arrival_slot": task.arrival_slot,
                         "destination": destination, "data_bits": task.data_bits,
                         "deadline_s": task.deadline_seconds,
                         "snapshot_path": list(old.action.path) if old.action else [],
                         "contact_path": list(new.action.path) if new.action else [],
                         "snapshot_contact_loss": prediction is not None and not prediction.topology_feasible,
                         "snapshot_unverified": prediction is not None and not prediction.fully_checked,
                         "contact_route_available": new.action is not None,
                         "contact_search_truncated": new.search_truncated})
    count = len(rows)
    summary = {"sampled_tasks": len(indices), "remote_destination_count": count,
               "snapshot_contact_loss_fraction": sum(r["snapshot_contact_loss"] for r in rows) / count if count else None,
               "snapshot_unverified_fraction": sum(r["snapshot_unverified"] for r in rows) / count if count else None,
               "contact_route_available_fraction": sum(r["contact_route_available"] for r in rows) / count if count else None,
               "note": "Evenly spaced tasks; reachable destinations audited individually in an empty system. "
                       "This diagnostic does not feed route alternatives to PPO and is not measured route failure."}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--max-tasks", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    if args.max_tasks < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Use positive max-tasks and distinct nonnegative seeds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is nonempty; use a new directory")
    config = load_config(args.config, args.set)
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
        current = load_config(args.config, args.set + ["simulation.seed=%s" % seed])
        tasks, capacities = generate_task_trace(current), generate_cpu_capacities(current)
        calibration = audit_scenario(current, topology, tasks, capacities)
        exposure, rows = inspect_routes(current, topology, tasks, capacities, args.max_tasks)
        directory = args.output / ("seed_%s" % seed)
        save_json(directory / "calibration.json", calibration)
        save_json(directory / "route_exposure.json", exposure)
        save_csv(directory / "destination_route_exposure.csv", rows)
        summary = {"seed": seed, "global_cpu_rho": calibration["expected_global_cpu_rho"],
                   "max_local_cpu_rho": max(calibration["expected_local_rho_per_satellite"]),
                   "optimistic_deadline_impossible_fraction": calibration["optimistic_intrinsic_deadline_impossible_fraction"],
                   "connected_slot_fraction": calibration["topology_diagnostics"]["connected_slot_fraction"],
                   "edge_state_changes": calibration["topology_diagnostics"]["edge_state_changes"],
                   "link_removal_window_fraction": calibration["current_link_removal_window_fraction"], **exposure}
        summaries.append(summary)
        save_csv(args.output / "summary.csv", summaries)
        print("seed=%s global_rho=%.3f max_local_rho=%.3f connected=%.3f snapshot_contact_loss=%s" %
              (seed, summary["global_cpu_rho"], summary["max_local_cpu_rho"],
               summary["connected_slot_fraction"], summary["snapshot_contact_loss_fraction"]), flush=True)
    print("Preflight diagnostics:", args.output.resolve())


if __name__ == "__main__":
    main()
