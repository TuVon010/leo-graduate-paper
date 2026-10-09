"""Compare scalar and packed PPO scoring on exactly the same frozen rollout.

This is a speed/correctness benchmark, not a learning-quality experiment. Both
paths use the same model weights, tasks, masks, joint likelihood and critic.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.env.leo_env import LeoEnv
from leo_routing.utils.io import save_json


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(config, encoder, device, steps, repeats):
    config = deepcopy(config)
    config["simulation"].update(slots=steps, drain_slots=max(40, config["routing"]["lookahead_slots"] + 1))
    config["rl"].update(encoder=encoder, device=device)
    agent = PPOAgent(config)
    env = LeoEnv(agent.configure_environment(config))
    observation, _ = env.reset()
    batches = []
    started = perf_counter()
    for _ in range(steps):
        actions, batch, _, _ = agent.choose_action(observation)
        batches.append(batch)
        observation, *_ = env.step(actions)
    collection = perf_counter() - started
    synchronize(agent.device)
    started = perf_counter()
    packed = agent.model.prepare_rollout(batches)
    synchronize(agent.device)
    packing = perf_counter() - started
    indices = np.arange(len(batches))

    def scalar():
        results = [agent.model.evaluate_batch(batch) for batch in batches]
        return tuple(torch.stack([r[j] for r in results]) for j in range(3))

    def batched():
        return agent.model.evaluate_minibatch(packed, indices)

    with torch.no_grad():
        old, new = scalar(), batched()
        errors = [float((a - b).abs().max()) for a, b in zip(old, new)]
        if not all(torch.allclose(a, b, atol=2e-4, rtol=2e-5) for a, b in zip(old, new)):
            raise AssertionError("Scalar/packed output mismatch: %s" % errors)

    def measure(evaluate):
        durations = []
        for iteration in range(repeats + 1):
            agent.model.zero_grad(set_to_none=True)
            synchronize(agent.device)
            started = perf_counter()
            logs, entropy, values = evaluate()
            # Exercise both actor and critic backward paths with bounded targets.
            loss = -logs.mean() - .002 * entropy.mean() + (values / 100).square().mean()
            loss.backward()
            synchronize(agent.device)
            if iteration:
                durations.append(perf_counter() - started)
        return float(np.median(durations))

    scalar_seconds, packed_seconds = measure(scalar), measure(batched)
    return dict(encoder=encoder, device=str(agent.device), gpu=torch.cuda.get_device_name(agent.device)
                if agent.device.type == "cuda" else None, steps=steps,
                rollout_device=str(agent.rollout_model.device),
                tasks=sum(len(b.decisions) for b in batches), repeats=repeats,
                collection_seconds=collection, packing_seconds=packing,
                scalar_forward_backward_seconds=scalar_seconds,
                packed_forward_backward_seconds=packed_seconds,
                scoring_speedup=scalar_seconds / packed_seconds, max_absolute_output_errors=errors,
                note="One frozen minibatch forward/backward; not total training speed or convergence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/coupled24.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--encoders", nargs="+", choices=["gat", "mlp"], default=["gat", "mlp"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    if args.output and args.output.exists():
        parser.error("Output exists; preserve it and choose a new file")
    config = load_config(args.config)
    rows = []
    for encoder in args.encoders:
        row = benchmark(config, encoder, args.device, args.steps, args.repeats)
        rows.append(row)
        print("[BENCHMARK] %s device=%s tasks=%s | scalar=%.4fs packed=%.4fs scoring_speedup=%.2fx pack=%.4fs" % (
            encoder, row["device"], row["tasks"], row["scalar_forward_backward_seconds"],
            row["packed_forward_backward_seconds"], row["scoring_speedup"], row["packing_seconds"]), flush=True)
    if args.output:
        save_json(args.output, rows)
        print("[FILES]", args.output.resolve())


if __name__ == "__main__":
    main()
