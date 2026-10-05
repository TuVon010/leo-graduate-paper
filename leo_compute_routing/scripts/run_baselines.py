"""Run algorithms on the same topology, task arrivals and CPU capacities."""
import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.baselines import POLICY_NAMES
from leo_routing.config import load_config
from leo_routing.evaluation.evaluator import run_comparison
from leo_routing.tasks.task_generator import load_task_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--algorithms", nargs="+", choices=POLICY_NAMES, default=list(POLICY_NAMES))
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--topology-cache", type=Path)
    parser.add_argument("--task-trace", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    output = args.output or PROJECT_ROOT / "results" / ("baselines_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    trace = load_task_trace(args.task_trace) if args.task_trace else None
    run_comparison(config, args.algorithms, output, args.topology_cache, trace, progress=lambda s: print(s, flush=True))
    print("Results:", output.resolve())


if __name__ == "__main__":
    main()
