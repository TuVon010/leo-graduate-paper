from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from ..models.actor_critic import ActorCritic
from ..config import fingerprint
from .features import BatchInput, FeatureBuilder, FEATURE_SCHEMA
from ..routing.reservations import ReservationCalendar
from torch.distributions import Categorical
from .settings import rl_settings, configure_rl_environment


class PPOAgent:
    """Autoregressive batch policy with one JOINT PPO ratio per physical slot."""
    def __init__(self, config, device=None):
        self.config = deepcopy(config)
        self.settings = rl_settings(config)
        seed = self.settings["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(self.settings["torch_threads"])
        selected_device = device or self.settings["device"]
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(selected_device)
        self.model = ActorCritic(self.settings).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings["learning_rate"], eps=1e-5)
        self.features = FeatureBuilder(config, self.settings)
        self.rng = np.random.default_rng(seed)
        self.update_count = 0
        self.deterministic = True
        self.use_future = self.settings["use_future"]
        self.name = self.settings["encoder"] + "_ppo"
        self.metadata = {"torch": str(torch.__version__), "encoder": self.settings["encoder"],
                         "training_config_sha256": fingerprint(config), "use_future": self.use_future,
                         "use_mask": self.settings["use_mask"], "use_reservations": self.settings["use_reservations"],
                         "shield_mode": self.settings["shield_mode"], "feature_schema": FEATURE_SCHEMA,
                         "candidate_generation": config["routing"].get("candidate_generation", "ksp"),
                         "reservation_is_guarantee": False}

    def configure_environment(self, config):
        configured = configure_rl_environment(config, self.settings)
        configured["routing"]["candidate_generation"] = self.config["routing"].get("candidate_generation", "ksp")
        return configured

    @torch.no_grad()
    def choose_action(self, observation, deterministic=False):
        self.model.eval()
        state = self.features.graph_input(observation)
        encoded = self.model.encode(state)
        log_probability = encoded[2] * 0
        decisions, chosen, actions = [], [], {}
        reserved_cpu = np.zeros_like(observation.cpu_capacities)
        reserved_links = {}
        calendar = ReservationCalendar.from_observation(observation)
        diagnostics = {"decision_count": 0, "predicted_invalid_selection_count": 0,
                       "fallback_count": 0, "candidate_count": 0, "predicted_invalid_candidate_count": 0,
                       "masked_candidate_count": 0, "blocked_probability_mass_sum": 0.0,
                       "search_truncated_task_count": 0}
        for position, task in enumerate(sorted(observation.tasks, key=lambda t: (t.deadline_seconds, t.task_id))):
            decision = self.features.decision_input(observation, task, position, reserved_cpu, reserved_links, calendar)
            logits = self.model.logits(encoded, decision)
            allowed = self.model.tensor(decision.mask, torch.bool)
            distribution = Categorical(logits=logits.masked_fill(~allowed, -torch.inf))
            choice = torch.argmax(distribution.logits) if deterministic else distribution.sample()
            index = int(choice.item())
            actions[task.task_id] = index
            log_probability += distribution.log_prob(choice)
            decisions.append(decision)
            chosen.append(index)
            self.features.book(observation.candidates[task.task_id][index].action, task, reserved_cpu, reserved_links, calendar)
            diagnostics["decision_count"] += 1
            diagnostics["predicted_invalid_selection_count"] += int(not decision.predicted_feasible[index])
            diagnostics["fallback_count"] += int(decision.fallback)
            diagnostics["candidate_count"] += len(decision.mask)
            diagnostics["predicted_invalid_candidate_count"] += int((~decision.predicted_feasible).sum())
            diagnostics["masked_candidate_count"] += int((~decision.mask).sum())
            diagnostics["blocked_probability_mass_sum"] += float(torch.softmax(logits, -1)[~allowed].sum())
            diagnostics["search_truncated_task_count"] += int(any(c.search_truncated for c in observation.candidates[task.task_id]))
        self.last_decision_metrics = diagnostics
        return actions, BatchInput(state, tuple(decisions), tuple(chosen)), float(log_probability.item()), float(encoded[2].item())

    def select(self, observation):
        return self.choose_action(observation, self.deterministic)[0]

    @torch.no_grad()
    def value(self, observation):
        self.model.eval()
        return float(self.model.encode(self.features.graph_input(observation))[2].item())

    def update(self, buffer):
        if not len(buffer):
            raise ValueError("Cannot update PPO with an empty rollout")
        self.model.train()
        advantages, returns = buffer.targets(self.settings)
        policy_steps = np.asarray([t.batch.has_choice for t in buffer.transitions], dtype=bool)
        if policy_steps.any():
            selected = advantages[policy_steps]
            advantages = (advantages - selected.mean()) / max(float(selected.std()), 1e-6)
        statistics = []
        early_stop = False
        for _ in range(self.settings["epochs"]):
            for start in range(0, len(buffer), self.settings["minibatch_steps"]):
                # One shuffle per epoch is built below; batches contain physical steps, not individual tasks.
                if start == 0:
                    order = self.rng.permutation(len(buffer))
                indices = order[start:start + self.settings["minibatch_steps"]]
                evaluated = [self.model.evaluate_batch(buffer.transitions[i].batch) for i in indices]
                logs, entropy, values = (torch.stack([result[j] for result in evaluated]) for j in range(3))
                old_logs = torch.tensor([buffer.transitions[i].log_probability for i in indices], device=self.device)
                targets = torch.as_tensor(returns[indices], device=self.device)
                advantage = torch.as_tensor(advantages[indices], device=self.device)
                valid = torch.as_tensor(policy_steps[indices], device=self.device)
                log_ratio = logs - old_logs
                ratio = torch.exp(log_ratio.clamp(-20, 20))
                if bool(valid.any()):
                    unclipped = ratio[valid] * advantage[valid]
                    clipped = ratio[valid].clamp(1 - self.settings["clip_ratio"], 1 + self.settings["clip_ratio"]) * advantage[valid]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    counts = torch.tensor([max(1, len(buffer.transitions[i].batch.decisions)) for i in indices], device=self.device)
                    entropy_bonus = (entropy[valid] / counts[valid]).mean()
                    approximate_kl = ((ratio[valid] - 1) - log_ratio[valid]).mean()
                    clip_fraction = ((ratio[valid] - 1).abs() > self.settings["clip_ratio"]).float().mean()
                else:
                    policy_loss = values.sum() * 0
                    entropy_bonus = policy_loss
                    approximate_kl = policy_loss
                    clip_fraction = policy_loss
                if not torch.isfinite(logs).all() or not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite PPO outputs")
                # Stop before applying an update to an already overly shifted batch policy.
                if bool(valid.any()) and float(approximate_kl.detach()) > self.settings["target_kl"]:
                    early_stop = True
                    break
                value_loss = F.mse_loss(values, targets)
                loss = (policy_loss + self.settings["value_coefficient"] * value_loss -
                        self.settings["entropy_coefficient"] * entropy_bonus)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite PPO objective")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings["max_grad_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                statistics.append({"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                                   "value_loss": float(value_loss.detach()), "entropy_per_task": float(entropy_bonus.detach()),
                                   "approximate_joint_kl": float(approximate_kl.detach()), "clip_fraction": float(clip_fraction.detach()),
                                   "gradient_norm": float(gradient_norm)})
            if early_stop:
                break
        self.update_count += 1
        self.model.eval()
        keys = ("loss", "policy_loss", "value_loss", "entropy_per_task", "approximate_joint_kl", "clip_fraction", "gradient_norm")
        metrics = {key: float(np.mean([row[key] for row in statistics])) if statistics else 0.0 for key in keys}
        predictions = np.asarray([t.value for t in buffer.transitions])
        variance = float(np.var(returns))
        metrics.update({"optimizer_steps": len(statistics), "kl_early_stop": early_stop,
                        "rollout_steps": len(buffer), "policy_steps": int(policy_steps.sum()),
                        "explained_variance_before_update": 1 - float(np.var(returns - predictions)) / variance if variance else None})
        return metrics

    def save(self, path, training_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"checkpoint_schema": 1, "feature_schema": FEATURE_SCHEMA, "config": self.config,
                   "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "update_count": self.update_count, "rng_state": deepcopy(self.rng.bit_generator.state),
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   "training_state": training_state or {}}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device=None, restore_optimizer=False):
        # Only tensors and primitive containers are saved: no pickled model/environment objects.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("checkpoint_schema") != 1 or payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Unsupported checkpoint or feature schema; schema-1 checkpoints require the old code. "
                             "Retrain for contact features and schema-2 normalization.")
        agent = cls(payload["config"], device)
        agent.model.load_state_dict(payload["model"])
        agent.update_count = payload["update_count"]
        if restore_optimizer:
            agent.optimizer.load_state_dict(payload["optimizer"])
            agent.rng.bit_generator.state = payload["rng_state"]
            torch.set_rng_state(payload["torch_rng"])
            if agent.device.type == "cuda" and payload["cuda_rng"]:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        agent.model.eval()
        return agent, payload["training_state"]
