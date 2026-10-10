from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from leo_routing.agents.features import FeatureBuilder
from leo_routing.agents.settings import rl_settings
from leo_routing.config import load_config
from leo_routing.env.leo_env import LeoEnv
from leo_routing.resource.allocator import allocate_capacity
from leo_routing.tasks.task import Task


def test_kkt_proxy_distinguishes_equal_total_work_and_matches_instantaneous_allocation(tiny_config):
    config = deepcopy(tiny_config)
    config["rl"] = {"feature_set": "kkt"}
    obs, _ = LeoEnv(config).reset()
    capacities = np.full_like(obs.cpu_capacities, 100.0)
    task = Task(100, 0, 100.0, 1.0, 20.0, 0)
    builder = FeatureBuilder(config, rl_settings(config))
    results = []
    for sizes in ([100.0], [25.0] * 4):
        jobs = tuple(SimpleNamespace(stage="cpu", remaining_cycles=w, compute_sat=0) for w in sizes)
        observation = replace(obs, active_jobs=jobs, cpu_capacities=capacities,
                              cpu_queue_cycles=np.array([100.0] + [0.0] * (len(capacities) - 1)))
        decision = builder.decision_input(observation, task, 0, np.zeros_like(capacities))
        proxy = np.expm1(decision.destination_features[0, -1])
        allocation = allocate_capacity({**{i: w for i, w in enumerate(sizes)}, "new": task.total_cycles}, 100.0)
        assert proxy == pytest.approx(task.total_cycles / allocation["new"], rel=1e-6)
        results.append(decision.destination_features)
    assert np.array_equal(results[0][:, :7], results[1][:, :7])
    assert results[1][0, -1] > results[0][0, -1]


def test_inflight_work_is_separate_and_does_not_reduce_instantaneous_cpu_share(tiny_config):
    config = deepcopy(tiny_config)
    config["rl"] = {"feature_set": "kkt"}
    obs, _ = LeoEnv(config).reset()
    task = Task(100, 0, 100, 1, 20, 0)
    builder = FeatureBuilder(config, rl_settings(config))
    base = builder.decision_input(obs, task, 0, np.zeros_like(obs.cpu_capacities))
    inflight = replace(obs, active_jobs=(SimpleNamespace(stage="tx", remaining_cycles=100, compute_sat=0),))
    other = builder.decision_input(inflight, task, 0, np.zeros_like(obs.cpu_capacities))
    assert np.array_equal(base.mask, other.mask)
    assert base.destination_features[0, -1] == other.destination_features[0, -1]
    assert other.destination_features[0, 9] > base.destination_features[0, 9]


def test_prefix_sqrt_is_immutable_and_changes_competition_proxy(tiny_config):
    config = deepcopy(tiny_config)
    config["rl"] = {"feature_set": "kkt"}
    obs, _ = LeoEnv(config).reset()
    task = Task(100, 0, 100, 1, 20, 0)
    builder = FeatureBuilder(config, rl_settings(config))
    reserved = np.zeros_like(obs.cpu_capacities)
    first = builder.decision_input(obs, task, 0, reserved, reserved_sqrt=reserved.copy())
    roots = reserved.copy(); roots[0] = 10
    second = builder.decision_input(obs, task, 1, reserved, reserved_sqrt=roots)
    before = second.destination_features.copy(); roots[:] = 0
    assert np.array_equal(before, second.destination_features)
    assert not second.destination_features.flags.writeable
    assert second.destination_features[0, -1] > first.destination_features[0, -1]


def test_optional_rl_overrides_preserve_historical_config(tiny_config):
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs/experiments/coupled24.yaml"
    assert "feature_set" not in load_config(path)["rl"]
    assert load_config(path, ["rl.feature_set=kkt", "rl.encoder=gated_gat"])["rl"]["feature_set"] == "kkt"
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(path, ["rl.nonexistent=true"])
