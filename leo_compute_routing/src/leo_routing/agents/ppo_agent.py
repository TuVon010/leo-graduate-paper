from copy import deepcopy
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import networkx as nx
import torch
from torch.nn import functional as F

from ..models.actor_critic import ActorCritic
from ..config import fingerprint
from .features import BatchInput, FeatureBuilder, FEATURE_SCHEMA
from ..routing.reservations import ReservationCalendar
from ..routing.contact_aware_router import ContactAwareRouter
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
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Use --device auto/cpu in the server runner, "
                               "or --set rl.device=auto/cpu in train_ppo.py.")
        self.model = ActorCritic(self.settings).to(self.device)
        # A small graph's sequential decisions can be faster on CPU, while
        # minibatch backpropagation remains on CUDA. No independently trained head.
        self.rollout_model = (deepcopy(self.model).to("cpu").requires_grad_(False)
                              if self.settings["rollout_device"] == "cpu" and self.device.type != "cpu"
                              else self.model)
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
                         "action_space": "computing_satellite", "feature_schema": FEATURE_SCHEMA,
                         "value_scale": self.settings["value_scale"],
                         "rollout_device": str(self.rollout_model.device),
                         "router_mode": config["routing"].get("mode", "contact"),
                         "reservation_is_guarantee": False}

    def configure_environment(self, config):
        configured = configure_rl_environment(config, self.settings)
        configured["routing"]["mode"] = self.config["routing"].get("mode", "contact")
        return configured

    @torch.no_grad()
    def choose_action(self, observation, deterministic=False):
        model = self.rollout_model
        model.eval()
        state = self.features.graph_input(observation)
        encoded = model.encode(state)
        log_probability = encoded[2] * 0
        blocked_mass = encoded[2] * 0
        decisions, chosen, actions = [], [], {}
        reserved_cpu = np.zeros_like(observation.cpu_capacities)
        calendar = ReservationCalendar.from_observation(observation)
        router = ContactAwareRouter(observation.routing_settings)
        hop_distances = {}
        diagnostics = dict(decision_count=0, predicted_invalid_selection_count=0, fallback_count=0,
                           destination_count=0, masked_destination_count=0, blocked_probability_mass_sum=0.0,
                           route_search_truncated_count=0, router_fallback_count=0)
        for position, task in enumerate(sorted(observation.tasks, key=lambda t: (t.deadline_seconds, t.task_id))):
            if task.source_sat not in hop_distances:
                hop_distances[task.source_sat] = nx.single_source_shortest_path_length(
                    observation.graph, task.source_sat, cutoff=observation.max_compute_hops)
            decision = self.features.decision_input(observation, task, position, reserved_cpu,
                                                   hop_distances[task.source_sat])
            logits = model.logits(encoded, decision)
            allowed = model.tensor(decision.mask, torch.bool)
            distribution = Categorical(logits=logits.masked_fill(~allowed, -torch.inf))
            choice = torch.argmax(distribution.logits) if deterministic else distribution.sample()
            destination = int(choice.item())
            # PPO likelihood is ONLY for the selected satellite, not the route or fallback.
            result = router.resolve(observation, task, destination, calendar)
            actions[task.task_id] = result.action
            log_probability += distribution.log_prob(choice)
            decisions.append(decision)
            chosen.append(destination)
            if self.settings["use_reservations"]:
                reserved_cpu[result.action.compute_sat] += task.total_cycles
                calendar.commit(task, result.action, result.estimate)
            router_fallback = result.reason in ("no_verified_route", "predicted_deadline")
            diagnostics["decision_count"] += 1
            diagnostics["predicted_invalid_selection_count"] += int(router_fallback)
            diagnostics["fallback_count"] += int(decision.fallback or router_fallback)
            diagnostics["router_fallback_count"] += int(router_fallback)
            diagnostics["destination_count"] += len(decision.mask)
            diagnostics["masked_destination_count"] += int((~decision.mask).sum())
            blocked_mass += torch.softmax(logits, -1)[~allowed].sum()
            diagnostics["route_search_truncated_count"] += int(result.search_truncated)
        log_value, critic_value, mass = torch.stack((log_probability, encoded[2], blocked_mass)).cpu().tolist()
        diagnostics["blocked_probability_mass_sum"] = mass
        self.last_decision_metrics = diagnostics
        return actions, BatchInput(state, tuple(decisions), tuple(chosen)), log_value, critic_value

    def select(self, observation):
        return self.choose_action(observation, self.deterministic)[0]

    @torch.no_grad()
    def value(self, observation):
        self.rollout_model.eval()
        return float(self.rollout_model.encode(self.features.graph_input(observation))[2].item())

    def _sync_rollout_model(self):
        if self.rollout_model is not self.model:
            self.rollout_model.load_state_dict(self.model.state_dict())
        self.rollout_model.eval()

    def update(self, buffer, progress=None, log_interval_seconds=10.0):
        if not len(buffer):
            raise ValueError("Cannot update PPO with an empty rollout")
        self.model.train()
        advantages, returns = buffer.targets(self.settings)
        policy_steps = np.asarray([t.batch.has_choice for t in buffer.transitions], dtype=bool)
        if policy_steps.any():
            selected = advantages[policy_steps]
            advantages = (advantages - selected.mean()) / max(float(selected.std()), 1e-6)
        packed = self.model.prepare_rollout([t.batch for t in buffer.transitions])
        old_logs_all = torch.tensor([t.log_probability for t in buffer.transitions], device=self.device)
        targets_all = torch.as_tensor(returns, device=self.device)
        advantages_all = torch.as_tensor(advantages, device=self.device)
        valid_all = torch.as_tensor(policy_steps, device=self.device)
        counts_all = torch.tensor([max(1, len(t.batch.decisions)) for t in buffer.transitions], device=self.device)
        statistics = []
        early_stop = False
        last_report = perf_counter()
        for epoch in range(self.settings["epochs"]):
            for start in range(0, len(buffer), self.settings["minibatch_steps"]):
                # One shuffle per epoch is built below; batches contain physical steps, not individual tasks.
                if start == 0:
                    order = self.rng.permutation(len(buffer))
                indices = order[start:start + self.settings["minibatch_steps"]]
                logs, entropy, values = self.model.evaluate_minibatch(packed, indices)
                index_tensor = torch.as_tensor(indices, device=self.device)
                old_logs, targets = old_logs_all[index_tensor], targets_all[index_tensor]
                advantage, valid = advantages_all[index_tensor], valid_all[index_tensor]
                has_policy = bool(policy_steps[indices].any())
                log_ratio = logs - old_logs
                ratio = torch.exp(log_ratio.clamp(-20, 20))
                if has_policy:
                    unclipped = ratio[valid] * advantage[valid]
                    clipped = ratio[valid].clamp(1 - self.settings["clip_ratio"], 1 + self.settings["clip_ratio"]) * advantage[valid]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    counts = counts_all[index_tensor]
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
                if has_policy and float(approximate_kl.detach()) > self.settings["target_kl"]:
                    early_stop = True
                    break
                raw_value_loss = F.mse_loss(values, targets)
                value_loss = raw_value_loss / self.settings["value_scale"] ** 2
                loss = (policy_loss + self.settings["value_coefficient"] * value_loss -
                        self.settings["entropy_coefficient"] * entropy_bonus)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite PPO objective")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings["max_grad_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                names = ("loss", "policy_loss", "value_loss", "entropy_per_task", "approximate_joint_kl",
                         "clip_fraction", "gradient_norm", "raw_value_loss", "gradient_clip_scale")
                scale = (self.settings["max_grad_norm"] / gradient_norm.clamp_min(1e-12)).clamp_max(1)
                numbers = torch.stack((loss.detach(), policy_loss.detach(), value_loss.detach(),
                    entropy_bonus.detach(), approximate_kl.detach(), clip_fraction.detach(),
                    gradient_norm.detach(), raw_value_loss.detach(), scale.detach())).cpu().tolist()
                statistics.append(dict(zip(names, numbers)))
                now = perf_counter()
                if progress and now - last_report >= log_interval_seconds:
                    last_report = now
                    progress("[OPTIMIZE] epoch=%s/%s minibatch=%s/%s optimizer_steps=%s loss=%.4f KL=%.6f" % (
                        epoch + 1, self.settings["epochs"], start // self.settings["minibatch_steps"] + 1,
                        (len(buffer) + self.settings["minibatch_steps"] - 1) // self.settings["minibatch_steps"],
                        len(statistics), statistics[-1]["loss"], statistics[-1]["approximate_joint_kl"]))
            if early_stop:
                break
        self.update_count += 1
        self.model.eval()
        self._sync_rollout_model()
        keys = ("loss", "policy_loss", "value_loss", "raw_value_loss", "entropy_per_task", "approximate_joint_kl",
                "clip_fraction", "gradient_norm", "gradient_clip_scale")
        metrics = {key: float(np.mean([row[key] for row in statistics])) if statistics else 0.0 for key in keys}
        predictions = np.asarray([t.value for t in buffer.transitions])
        variance = float(np.var(returns))
        metrics.update({"optimizer_steps": len(statistics), "kl_early_stop": early_stop,
                        "rollout_steps": len(buffer), "policy_steps": int(policy_steps.sum()),
                        "value_prediction_mean": float(predictions.mean()), "value_prediction_std": float(predictions.std()),
                        "return_mean": float(returns.mean()), "return_std": float(returns.std()),
                        "value_scale": self.settings["value_scale"],
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
            raise ValueError("Unsupported checkpoint: old joint-path policies are incompatible. "
                             "Retrain the satellite-only policy with feature schema 3.")
        agent = cls(payload["config"], device)
        agent.model.load_state_dict(payload["model"])
        agent._sync_rollout_model()
        agent.update_count = payload["update_count"]
        if restore_optimizer:
            agent.optimizer.load_state_dict(payload["optimizer"])
            agent.rng.bit_generator.state = payload["rng_state"]
            torch.set_rng_state(payload["torch_rng"])
            if agent.device.type == "cuda" and payload["cuda_rng"]:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        agent.model.eval()
        return agent, payload["training_state"]
