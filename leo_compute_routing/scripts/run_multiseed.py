"""Preserve per-seed traces and produce seed-level paired bootstrap statistics."""
import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.baselines import POLICY_NAMES
from leo_routing.config import load_config
from leo_routing.evaluation.statistics import run_multiseed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--algorithms", nargs="+", choices=POLICY_NAMES, default=list(POLICY_NAMES))
    parser.add_argument("--reference", choices=POLICY_NAMES, default="batch_greedy")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7301)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    output = args.output or PROJECT_ROOT / "results" / ("multiseed_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_multiseed(config, args.algorithms, args.seeds, output, args.reference,
                  args.bootstrap_samples, args.bootstrap_seed, progress=lambda s: print(s, flush=True))
    print("Results:", output.resolve())


if __name__ == "__main__":
    main()
