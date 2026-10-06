"""Training-seed prefix probe of critic saturation and candidate rankings.

Does not evaluate complete episodes, select checkpoints or use test seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.agents.features import CONTACT_COMPLETION_INDEX
from leo_routing.env.leo_env import LeoEnv
from leo_routing.topology.topology_cache import TopologyTrace


def probe(root, output, slots, seed):
    report = {}
    for case in ("compute24", "contact66"):
        folder = root / "train" / case / "full/init_2026"
        for name in ("update_00010.pt", "update_00020.pt"):
            agent, _ = PPOAgent.load(folder / name, "cpu")
            if seed not in agent.settings["train_seeds"]:
                raise ValueError("Use a training seed for this diagnostic")
            config = agent.configure_environment(agent.config)
            config["simulation"]["seed"] = seed
            env = LeoEnv(config, TopologyTrace.load(folder / "topology.npz"))
            obs, _ = env.reset()
            values, rewards, gap, rank, local, saturation = [], [], [], [], [], []
            def record(module, inputs, result):
                saturation.append(float((result.abs() > .99).float().mean()))
            handle = agent.model.context_encoder[-1].register_forward_hook(record)
            steps = 0
            try:
                for _ in range(slots):
                    actions, batch, _, value = agent.choose_action(obs, deterministic=True)
                    values.append(value)
                    for decision, choice in zip(batch.decisions, batch.actions):
                        times = np.expm1(decision.candidates[:, CONTACT_COMPLETION_INDEX]) * agent.features.time_scale
                        allowed = times[decision.mask]
                        gap.append(float(times[choice] - allowed.min()))
                        rank.append(int(np.sum(allowed < times[choice] - 1e-6)))
                        local.append(choice == 0)
                    obs, reward, done, truncated, _ = env.step(actions)
                    rewards.append(reward)
                    steps += 1
                    if done or truncated:
                        break
            finally:
                handle.remove()
            row = {"seed": seed, "prefix_seconds": steps * config["simulation"]["slot_seconds"],
                   "value_mean": float(np.mean(values)), "value_std": float(np.std(values)),
                   "value_range": [min(values), max(values)], "reward_mean": float(np.mean(rewards)),
                   "context_saturation_fraction": float(np.mean(saturation)),
                   "chosen_completion_proxy_gap_s": float(np.mean(gap)) if gap else None,
                   "chosen_proxy_rank_mean_zero_is_best": float(np.mean(rank)) if rank else None,
                   "local_selection_fraction": float(np.mean(local)) if local else None}
            report[case + "/" + name] = row
            print(case, name, json.dumps(row), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.slots < 1:
        parser.error("slots must be positive")
    if args.output.exists():
        parser.error("Preserve the old probe; use a new output file")
    probe(args.root, args.output, args.slots, args.seed)
