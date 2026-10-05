"""Train candidate MLP-PPO or edge-aware GAT-PPO; no simulator logic here."""
import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/ppo.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    try:
        from leo_routing.agents.trainer import train_ppo
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("PyTorch is missing. Install requirements-rl.txt in this Python environment; see docs/RL.md.")
        raise
    config = load_config(args.config, args.set)
    output = args.output or PROJECT_ROOT / "results" / ("ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    train_ppo(config, output, args.resume, progress=lambda s: print(s, flush=True))
    print("Checkpoints and logs:", output.resolve())


if __name__ == "__main__":
    main()
