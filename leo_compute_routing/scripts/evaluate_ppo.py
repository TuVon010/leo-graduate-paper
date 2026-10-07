"""Compare frozen checkpoints; optionally label reused validation seeds as development evaluation."""
import argparse
import math
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.baselines import POLICY_NAMES
from leo_routing.config import load_config
from leo_routing.utils.console import capture_console, console_log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[201])
    parser.add_argument("--algorithms", nargs="*", choices=POLICY_NAMES,
                        default=["local", "shortest_offload", "least_load", "computing_aware", "computing_aware_future", "node_greedy"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-validation-reuse", action="store_true",
                        help="Allow validation seeds for debugging; output is NOT an independent test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.log_interval_seconds) or args.log_interval_seconds <= 0:
        parser.error("--log-interval-seconds must be positive and finite")
    output = args.output or PROJECT_ROOT / "results" / ("ppo_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    with capture_console(console_log_path(output, "evaluate")) as log:
        print("[FILES] evaluation_results=%s | console_log=%s" % (output.resolve(), log), flush=True)
        try:
            from leo_routing.evaluation.rl_evaluator import evaluate_checkpoints
        except ModuleNotFoundError as error:
            if error.name == "torch":
                parser.error("PyTorch is missing. See docs/RL.md and requirements-rl.txt.")
            raise
        evaluate_checkpoints(args.checkpoints, args.seeds, output, args.algorithms,
                             load_config(args.config) if args.config else None, args.device,
                             args.bootstrap_samples, progress=lambda s: print(s, flush=True),
                             log_interval_seconds=args.log_interval_seconds,
                             allow_validation_reuse=args.allow_validation_reuse)
        print("[EVAL COMPLETE] results=%s | console_log=%s" % (output.resolve(), log), flush=True)


if __name__ == "__main__":
    main()
