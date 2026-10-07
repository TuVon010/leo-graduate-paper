from copy import deepcopy

import numpy as np
import pytest

from leo_routing.baselines import POLICY_NAMES, make_policy
from leo_routing.env.leo_env import LeoEnv
from leo_routing.evaluation.evaluator import run_comparison
from leo_routing.tasks.task import Task
from leo_routing.tasks.task_generator import generate_task_trace


def test_reset_step_and_repeatability(tiny_config):
    environment = LeoEnv(tiny_config)
    def run():
        observation, _ = environment.reset()
        assert observation.node_features.shape == (6, 6)
        assert observation.edge_index.shape[0] == 2
        assert observation.edge_features.shape[1] == 4
        assert not observation.node_features.flags.writeable
        policy = make_policy("computing_aware")
        while True:
            observation, reward, terminated, truncated, info = environment.step(policy.select(observation))
            assert np.isfinite(reward)
            if terminated or truncated:
                assert observation is None
                return environment.engine.task_records(), info["episode_metrics"]
    assert run() == run()
    with pytest.raises(RuntimeError):
        environment.step({})


def test_bad_actions_do_not_mutate_environment(tiny_config):
    trace = ((Task(0, 0, 10, 1, 3, 0),), (), ())
    environment = LeoEnv(tiny_config, task_trace=trace)
    environment.reset()
    for actions in ({}, {0: -1}, {0: True}, {0: 9999}, {0: 0, 1: 0}):
        with pytest.raises(ValueError):
            environment.step(actions)
        assert not environment.engine.jobs and environment.slot == 0


def test_empty_arrivals_produce_finite_metrics(tiny_config, tmp_path):
    config = deepcopy(tiny_config)
    config["tasks"]["arrival_rate_per_slot"] = 0.0
    results = run_comparison(config, ["local"], tmp_path)
    assert results[0]["task_count"] == 0
    assert results[0]["mean_completion_delay_s"] is None
    assert results[0]["success_rate"] == 0
    assert results[0]["terminated"]


def test_censored_task_not_reported_as_completed(tiny_config):
    config = deepcopy(tiny_config)
    config["simulation"].update(slots=1, drain_slots=0)
    trace = ((Task(0, 0, 1e9, 1e3, 0.5, 0),),)
    environment = LeoEnv(config, task_trace=trace)
    observation, _ = environment.reset()
    _, _, terminated, truncated, info = environment.step({0: 0})
    metrics = info["episode_metrics"]
    assert truncated and not terminated
    assert metrics["completed_count"] == 0 and metrics["censored_rate"] == 1
    assert metrics["mean_completion_delay_s"] is None and metrics["deadline_violation_rate"] == 1


def test_all_baselines_share_exogenous_traces(tiny_config, tmp_path):
    rows = run_comparison(tiny_config, POLICY_NAMES, tmp_path)
    assert len({row["task_trace_sha256"] for row in rows}) == 1
    assert len({row["cpu_capacities_sha256"] for row in rows}) == 1
    assert len({row["topology_signature"] for row in rows}) == 1
    assert len({row["task_count"] for row in rows}) == 1
    assert all(row["max_cpu_budget_ratio"] <= 1 + 1e-12 for row in rows)
    assert all(row["max_link_budget_ratio"] <= 1 + 1e-12 for row in rows)


def test_task_trace_does_not_depend_on_policy(tiny_config):
    assert generate_task_trace(tiny_config) == generate_task_trace(tiny_config)
    other = deepcopy(tiny_config)
    other["simulation"]["seed"] += 1
    assert generate_task_trace(tiny_config) != generate_task_trace(other)


def test_future_ablation_preserves_deadline_filtering(tiny_config, tmp_path):
    import yaml
    from leo_routing.evaluation.ablation import run_ablation
    rows = run_ablation(tiny_config, tmp_path)
    with (tmp_path / "without_future_topology" / "resolved_config.yaml").open(encoding="utf-8") as stream:
        resolved = yaml.safe_load(stream)
    assert resolved["routing"]["lookahead_slots"] == 0
    assert resolved["routing"]["deadline_mask"] == tiny_config["routing"]["deadline_mask"]
    assert len({row["task_trace_sha256"] for row in rows}) == 1


def test_route_rejection_is_penalized_and_not_a_physical_failure(tiny_config):
    from leo_routing.routing.action_builder import RoutingAction
    task = Task(0, 0, 10, 1, 3, 0)
    rewards = []
    for rejected in (False, True):
        env = LeoEnv(tiny_config, task_trace=((task,), (), ()))
        env.reset()
        action = RoutingAction(0, 0, (0,), 1 if rejected else None,
                               "no_verified_route" if rejected else "")
        _, reward, _, _, info = env.step({0: action})
        rewards.append(reward)
        assert info["slot_metrics"]["new_route_failures"] == 0
        assert info["slot_metrics"]["new_routing_rejections"] == int(rejected)
    assert rewards[0] - rewards[1] == pytest.approx(
        tiny_config["reward"]["route_failure_penalty"] / tiny_config["reward"]["normalizer"])
