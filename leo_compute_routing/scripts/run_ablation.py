import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.evaluation.ablation import run_ablation


def main():
    parser = argparse.ArgumentParser(description="Run simulator/heuristic ablations")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/smoke.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/ablation")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    run_ablation(load_config(args.config, args.set), args.output, progress=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
