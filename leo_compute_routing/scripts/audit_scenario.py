"""Inspect offered load, intrinsic deadline bounds and contact observability."""
import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.env.leo_env import generate_cpu_capacities
from leo_routing.evaluation.calibration import audit_scenario
from leo_routing.tasks.task_generator import generate_task_trace
from leo_routing.topology.topology_cache import get_topology
from leo_routing.utils.io import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/scenario_audit.json")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    report = audit_scenario(config, get_topology(config), generate_task_trace(config), generate_cpu_capacities(config))
    save_json(args.output, report)
    print("Hotspot task fraction:", report["expected_hotspot_task_fraction"])
    print("Max expected local rho:", max(report["expected_local_rho_per_satellite"]))
    print("Global expected rho:", report["expected_global_cpu_rho"])
    print("Intrinsic impossible fraction:", report["optimistic_intrinsic_deadline_impossible_fraction"])
    print("Prediction-window link removal fraction:", report["current_link_removal_window_fraction"])
    for warning in report["warnings"]:
        print("Calibration:", warning)
    print("Report:", args.output.resolve())


if __name__ == "__main__":
    main()
