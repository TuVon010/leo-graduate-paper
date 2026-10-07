import argparse
from pathlib import Path

import yaml

from _bootstrap import PROJECT_ROOT
from leo_routing.baselines import POLICY_NAMES
from leo_routing.config import load_config
from leo_routing.evaluation.sensitivity import run_sensitivity


def main():
    parser = argparse.ArgumentParser(description="Replay parameter sweeps for multiple seeds")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/sensitivity")
    parser.add_argument("--parameter", default="tasks.arrival_rate_per_slot")
    parser.add_argument("--values", nargs="+", default=["4", "8", "12", "20"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100])
    parser.add_argument("--algorithms", nargs="+", choices=POLICY_NAMES,
                        default=["local", "computing_aware", "computing_aware_future"])
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    run_sensitivity(load_config(args.config, args.set), args.parameter, [yaml.safe_load(v) for v in args.values],
                    args.seeds, args.algorithms, args.output, progress=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
