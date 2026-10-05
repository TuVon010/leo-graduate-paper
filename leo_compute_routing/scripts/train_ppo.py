"""Train candidate MLP-PPO or edge-aware GAT-PPO; no simulator logic here."""
import argparse
import math
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.utils.console import capture_console, console_log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/ppo.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.log_interval_seconds) or args.log_interval_seconds <= 0:
        parser.error("--log-interval-seconds must be positive and finite")
    output = args.output or PROJECT_ROOT / "results" / ("ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    with capture_console(console_log_path(output, "train")) as log:
        try:
            from leo_routing.agents.trainer import train_ppo
        except ModuleNotFoundError as error:
            if error.name == "torch":
                parser.error("PyTorch is missing. Install requirements-rl.txt in this Python environment; see docs/RL.md.")
            raise
        config = load_config(args.config, args.set)
        train_ppo(config, output, args.resume, progress=lambda s: print(s, flush=True),
                  log_interval_seconds=args.log_interval_seconds, console_log=log)
        print("[FILES] results=%s | console_log=%s" % (output.resolve(), log), flush=True)


if __name__ == "__main__":
    main()
