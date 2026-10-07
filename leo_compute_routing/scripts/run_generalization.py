"""Evaluate frozen satellite-only policies on named unseen constellations."""
import argparse
import math
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.baselines import POLICY_NAMES
from leo_routing.utils.console import capture_console, console_log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[301])
    parser.add_argument("--algorithms", nargs="*", choices=POLICY_NAMES, default=["local", "computing_aware", "computing_aware_future"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.log_interval_seconds) or args.log_interval_seconds <= 0:
        parser.error("--log-interval-seconds must be positive and finite")
    names = [path.stem for path in args.configs]
    if len(set(names)) != len(names):
        parser.error("Target config filenames must have distinct stems")
    output = args.output or PROJECT_ROOT / "results" / ("generalization_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    with capture_console(console_log_path(output, "generalization")) as log:
        print("[FILES] generalization_results=%s | console_log=%s" % (output.resolve(), log), flush=True)
        from leo_routing.evaluation.generalization import run_generalization
        run_generalization(args.checkpoints, {p.stem: load_config(p) for p in args.configs}, args.seeds, output,
                           args.algorithms, args.device, args.bootstrap_samples, progress=lambda s: print(s, flush=True),
                           log_interval_seconds=args.log_interval_seconds)
        print("[GENERALIZATION COMPLETE] results=%s | console_log=%s" % (output.resolve(), log), flush=True)


if __name__ == "__main__":
    main()
