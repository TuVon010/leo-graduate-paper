from copy import deepcopy
import math

DEFAULTS = {
    "encoder": "gat", "hidden_dim": 64, "gat_heads": 4, "gat_layers": 2,
    "use_future": True, "use_mask": True, "use_reservations": True, "time_scale_seconds": 1.0,
    "value_scale": 100.0,
    "learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.95, "clip_ratio": 0.2,
    "value_coefficient": 0.5, "entropy_coefficient": 0.01, "max_grad_norm": 0.5,
    "epochs": 4, "minibatch_steps": 16, "target_kl": 0.03,
    "updates": 200, "episodes_per_update": 2, "seed": 2026, "device": "auto", "rollout_device": "same", "torch_threads": 1,
    "train_seeds": [2026], "validation_seeds": [100],
    "validation_every": 5, "checkpoint_every": 10,
}


def rl_settings(config):
    supplied = config.get("rl", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown RL configuration fields: %s" % (set(supplied) - set(DEFAULTS) if isinstance(supplied, dict) else supplied))
    settings = {**deepcopy(DEFAULTS), **deepcopy(supplied)}
    if settings["encoder"] not in ("mlp", "gat"):
        raise ValueError("rl.encoder must be mlp or gat")
    if settings["rollout_device"] not in ("same", "cpu"):
        raise ValueError("rl.rollout_device must be same or cpu")
    for key in ("hidden_dim", "gat_heads", "gat_layers", "epochs", "minibatch_steps", "updates",
                "episodes_per_update", "torch_threads", "validation_every", "checkpoint_every"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("rl.%s must be a positive integer" % key)
    if settings["hidden_dim"] % settings["gat_heads"]:
        raise ValueError("hidden_dim must be divisible by gat_heads")
    if isinstance(settings["seed"], bool) or not isinstance(settings["seed"], int) or settings["seed"] < 0:
        raise ValueError("rl.seed must be a nonnegative integer")
    for key in ("use_future", "use_mask", "use_reservations"):
        if not isinstance(settings[key], bool):
            raise ValueError("rl.%s must be boolean" % key)
    for key in ("time_scale_seconds", "learning_rate", "gamma", "gae_lambda", "clip_ratio",
                "value_coefficient", "max_grad_norm", "target_kl", "value_scale"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("rl.%s must be positive and finite" % key)
    if settings["gamma"] > 1 or settings["gae_lambda"] > 1 or settings["clip_ratio"] >= 1:
        raise ValueError("Invalid discount, GAE or clipping coefficient")
    entropy = settings["entropy_coefficient"]
    if isinstance(entropy, bool) or not isinstance(entropy, (int, float)) or not math.isfinite(entropy) or entropy < 0:
        raise ValueError("Invalid entropy_coefficient")
    for key in ("train_seeds", "validation_seeds"):
        values = settings[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ValueError("rl.%s must contain distinct nonnegative seeds" % key)
    if set(settings["train_seeds"]) & set(settings["validation_seeds"]):
        raise ValueError("Training and validation seeds must be disjoint")
    if not isinstance(settings["device"], str):
        raise ValueError("rl.device must be auto, cpu or a Torch device string")
    return settings


def configure_rl_environment(config, settings):
    configured = deepcopy(config)
    if not settings["use_future"]:
        configured["routing"]["lookahead_slots"] = 0
    return configured
