"""Reproducible episode diversity, independent of policy RNG and encoder.

Random access by episode index makes checkpoint continuation exact without
resetting a mutable stream. One root seed defines the whole training sequence.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np

from ..config import fingerprint
from ..env.leo_env import generate_cpu_capacities
from ..tasks.task_generator import generate_task_trace
from ..topology.walker import EARTH_RADIUS_M, EARTH_MU

SCENARIO_SCHEMA = 1


def episode_configuration(config, settings, episode_index):
    """Return configuration and provenance for a zero-based training episode."""
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("Episode index must be a nonnegative integer")
    configured = deepcopy(config)
    root = settings["train_seeds"][episode_index % len(settings["train_seeds"])]
    seed = root
    if settings["randomize_episodes"]:
        # Separate domains for arrivals/CPU (already split inside their generators),
        # orbital epoch and hotspot identities. No consumption of Torch/global RNG.
        retry = 0
        blocked = set(settings["validation_seeds"]) | set(settings["train_seeds"])
        while seed in blocked:
            seed = int(np.random.SeedSequence([root, episode_index, 3101, retry])
                       .generate_state(1, dtype=np.uint64)[0])
            retry += 1
        rng = np.random.default_rng(np.random.SeedSequence([root, episode_index, 3102]))
        if configured["topology"]["mode"] == "walker":
            radius = EARTH_RADIUS_M + configured["topology"]["altitude_m"]
            period = 2 * math.pi * math.sqrt(radius ** 3 / EARTH_MU)
            configured["topology"]["epoch_offset_seconds"] = float(rng.uniform(0.0, period))
        hotspot_rng = np.random.default_rng(np.random.SeedSequence([root, episode_index, 3103]))
        topo = configured["topology"]
        count = topo["planes"] * topo["sats_per_plane"]
        hot_count = len(configured["tasks"]["hotspot_satellites"])
        configured["tasks"]["hotspot_satellites"] = sorted(
            int(s) for s in hotspot_rng.choice(count, hot_count, replace=False))
    configured["simulation"]["seed"] = seed
    provenance = dict(scenario_schema=SCENARIO_SCHEMA, episode_index=episode_index,
                      root_seed=root, episode_seed=seed,
                      randomize_episodes=settings["randomize_episodes"],
                      epoch_offset_seconds=configured["topology"].get("epoch_offset_seconds", 0.0),
                      hotspot_satellites=configured["tasks"]["hotspot_satellites"])
    return configured, provenance


def episode_inputs(config):
    """Generate shared exogenous inputs and hashes for audit/replay."""
    tasks = generate_task_trace(config)
    capacities = generate_cpu_capacities(config)
    return tasks, capacities, dict(
        task_trace_sha256=fingerprint([[asdict(t) for t in batch] for batch in tasks]),
        cpu_capacities_sha256=fingerprint(capacities.tolist()))
