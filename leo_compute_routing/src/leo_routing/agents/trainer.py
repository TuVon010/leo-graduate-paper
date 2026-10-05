"""Complete-episode on-policy collection, validation-only checkpoint selection."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import platform

import numpy as np
import torch

from ..config import fingerprint
from ..env.leo_env import LeoEnv, generate_cpu_capacities
from ..evaluation.evaluator import evaluate_policy
from ..tasks.task_generator import generate_task_trace
from ..topology.topology_cache import generate_topology
from ..utils.io import save_csv, save_json, save_yaml
from .ppo_agent import PPOAgent
from .rollout_buffer import RolloutBuffer, Transition


def train_ppo(config, output_directory, resume=None, progress=None):
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output is nonempty; use a new directory")
    agent, state = PPOAgent.load(resume, config.get("rl", {}).get("device"), True) if resume else (PPOAgent(config), {})
    if resume and fingerprint(agent.config) != fingerprint(config):
        # Device and target update count may change, but architecture/scales/seeds/objective may not.
        old, new = deepcopy(agent.config), deepcopy(config)
        for resolved in (old, new):
            for field in ("device", "updates"):
                resolved.get("rl", {}).pop(field, None)
        if fingerprint(old) != fingerprint(new):
            raise ValueError("Resume configuration changed beyond device/updates")
        agent.settings["updates"] = config.get("rl", {}).get("updates", agent.settings["updates"])
        agent.config = deepcopy(config)
    settings = agent.settings
    if settings["updates"] <= agent.update_count:
        raise ValueError("Target updates must exceed the checkpoint update count")
    configured = agent.configure_environment(config)
    topology = generate_topology(configured)
    output.mkdir(parents=True, exist_ok=True)
    save_yaml(output / "resolved_config.yaml", config)
    save_json(output / "training_manifest.json", {"schema": 1, "config_sha256": fingerprint(config),
              "torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
              "device": str(agent.device), "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(agent.device) if agent.device.type == "cuda" else None,
              "resume_from": str(Path(resume).resolve()) if resume else None,
              "train_seeds": settings["train_seeds"], "validation_seeds": settings["validation_seeds"],
              "selection": "maximize mean validation episode reward; never test-set performance",
              "ppo_action": "autoregressive batch; summed conditional log probabilities; one physical-slot ratio",
              "truncation": "environment drain censorship is a terminal finite-horizon outcome, bootstrap=0"})
    topology.save(output / "topology.npz")
    episode_count = state.get("episode_count", 0)
    best_reward = state.get("best_validation_reward", None)
    best_update = state.get("best_update", None)
    if resume:
        # This keeps the previous best available in a new resume directory.
        old_best = Path(resume).parent / "best.pt"
        old_best_update = (torch.load(old_best, map_location="cpu", weights_only=True).get("update_count")
                           if old_best.exists() else None)
        if old_best.exists() and old_best_update == best_update:
            from shutil import copy2
            copy2(old_best, output / "best.pt")
        else:
            # A historical checkpoint may predate the source directory's best.
            # Do not carry a model trained after the resumed checkpoint forward.
            best_reward, best_update = None, None
    episodes, updates, validation = [], [], []
    started = perf_counter()
    while agent.update_count < settings["updates"]:
        buffer = RolloutBuffer()
        next_update = agent.update_count + 1
        for _ in range(settings["episodes_per_update"]):
            seed = settings["train_seeds"][episode_count % len(settings["train_seeds"])]
            episode_config = deepcopy(configured)
            episode_config["simulation"]["seed"] = seed
            environment = LeoEnv(episode_config, topology)
            observation, _ = environment.reset()
            fallback_count, choices = 0, 0
            while True:
                actions, batch, log_probability, value = agent.choose_action(observation)
                next_observation, reward, terminated, truncated, info = environment.step(actions)
                terminal = terminated or truncated
                # Next value is only needed at rollout cutoffs; full episodes
                # allow it to be filled from the following sampled state's value.
                buffer.append(Transition(batch, log_probability, value, reward, 0.0, terminal))
                if len(buffer.transitions) > 1 and not buffer.transitions[-2].terminal:
                    previous = buffer.transitions[-2]
                    buffer.transitions[-2] = Transition(previous.batch, previous.log_probability, previous.value,
                                                        previous.reward, value, False)
                fallback_count += sum(d.fallback for d in batch.decisions)
                choices += len(batch.decisions)
                if terminal:
                    break
                observation = next_observation
            episode_count += 1
            episodes.append({"episode": episode_count, "update": next_update, "seed": seed,
                             "fallback_count": fallback_count, "decision_count": choices, **info["episode_metrics"]})
            if progress:
                progress("Episode %s seed=%s reward=%.4f success=%.3f" %
                         (episode_count, seed, info["episode_metrics"]["total_reward"], info["episode_metrics"]["success_rate"]))
        metrics = agent.update(buffer)
        update = agent.update_count
        updates.append({"update": update, "elapsed_seconds": perf_counter() - started, **metrics})
        save_csv(output / "episodes.csv", episodes)
        save_csv(output / "updates.csv", updates)
        if progress:
            progress("Update %s loss=%.4f policy_steps=%s optimizer_steps=%s" %
                     (update, metrics["loss"], metrics["policy_steps"], metrics["optimizer_steps"]))
        if update % settings["validation_every"] == 0 or update == settings["updates"]:
            rewards = []
            for seed in settings["validation_seeds"]:
                validation_config = deepcopy(configured)
                validation_config["simulation"]["seed"] = seed
                row, _ = evaluate_policy(validation_config, topology, generate_task_trace(validation_config),
                                         generate_cpu_capacities(validation_config), agent)
                rewards.append(row["total_reward"])
                validation.append({"update": update, **row})
            average_reward = float(np.mean(rewards))
            if best_reward is None or average_reward > best_reward:
                best_reward, best_update = average_reward, update
                agent.save(output / "best.pt", {"episode_count": episode_count,
                           "best_validation_reward": best_reward, "best_update": best_update})
            save_csv(output / "validation.csv", validation)
            if progress:
                progress("Validation reward=%.4f; best update=%s" % (average_reward, best_update))
        training_state = {"episode_count": episode_count, "best_validation_reward": best_reward, "best_update": best_update}
        agent.save(output / "last.pt", training_state)
        if update % settings["checkpoint_every"] == 0:
            agent.save(output / ("update_%05d.pt" % update), training_state)
    save_json(output / "training_summary.json", {"updates": agent.update_count, "episodes": episode_count,
              "best_update": best_update, "best_validation_reward": best_reward,
              "elapsed_seconds": perf_counter() - started, "test_set_evaluated": False})
    return agent
