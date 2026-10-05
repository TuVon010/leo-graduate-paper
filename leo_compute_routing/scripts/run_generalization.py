"""Evaluate frozen variable-candidate policies on named unseen constellations."""
import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.baselines import POLICY_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203, 204, 205])
    parser.add_argument("--algorithms", nargs="*", choices=POLICY_NAMES, default=["local", "computing_aware", "computing_aware_future"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = [path.stem for path in args.configs]
    if len(set(names)) != len(names):
        parser.error("Target config filenames must have distinct stems")
    from leo_routing.evaluation.generalization import run_generalization
    output = args.output or PROJECT_ROOT / "results" / ("generalization_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_generalization(args.checkpoints, {p.stem: load_config(p) for p in args.configs}, args.seeds, output,
                       args.algorithms, args.device, args.bootstrap_samples, progress=lambda s: print(s, flush=True))
    print("Frozen-policy study:", output.resolve())


if __name__ == "__main__":
    main()
