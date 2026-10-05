"""Compare frozen MLP/GAT checkpoints and existing baselines on held-out seeds."""
import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.baselines import POLICY_NAMES
from leo_routing.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[201, 202, 203, 204, 205])
    parser.add_argument("--algorithms", nargs="*", choices=POLICY_NAMES,
                        default=["local", "shortest_offload", "least_load", "computing_aware", "computing_aware_future"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    try:
        from leo_routing.evaluation.rl_evaluator import evaluate_checkpoints
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("PyTorch is missing. See docs/RL.md and requirements-rl.txt.")
        raise
    output = args.output or PROJECT_ROOT / "results" / ("ppo_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    evaluate_checkpoints(args.checkpoints, args.seeds, output, args.algorithms,
                         load_config(args.config) if args.config else None, args.device,
                         args.bootstrap_samples, progress=lambda s: print(s, flush=True))
    print("Evaluation results:", output.resolve())


if __name__ == "__main__":
    main()
