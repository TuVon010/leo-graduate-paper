from copy import deepcopy

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.agents.rollout_buffer import RolloutBuffer, Transition
from leo_routing.agents.settings import rl_settings
from leo_routing.baselines import make_policy
from leo_routing.env.leo_env import LeoEnv
from leo_routing.evaluation.evaluator import run_comparison
from leo_routing.tasks.task import Task


def stabilized(tiny_config):
    config = deepcopy(tiny_config)
    config["routing"].update(lookahead_slots=5)
    config["rl"] = dict(encoder="gat", hidden_dim=16, device="cpu",
                        value_scale=100.0, epochs=1, minibatch_steps=64)
    return config


def test_scaled_critic_keeps_reward_units_and_checkpoint_roundtrip(tiny_config, tmp_path):
    config = stabilized(tiny_config)
    agent = PPOAgent(config)
    env = LeoEnv(config)
    obs, _ = env.reset()
    buffer = RolloutBuffer()
    while True:
        actions, batch, log, value = agent.choose_action(obs)
        encoded = agent.model.encode(batch.graph)
        assert value == pytest.approx(float(agent.model.value_network(encoded[1]).detach()) * 100, abs=1e-5)
        obs, reward, done, truncated, _ = env.step(actions)
        terminal = done or truncated
        buffer.append(Transition(batch, log, value, reward, 0 if terminal else agent.value(obs), terminal))
        if terminal:
            break
    metrics = agent.update(buffer)
    assert metrics["raw_value_loss"] == pytest.approx(metrics["value_loss"] * 10000, rel=1e-5)
    assert metrics["return_std"] >= 0 and 0 < metrics["gradient_clip_scale"] <= 1
    assert np.isfinite([metrics["return_mean"], metrics["value_prediction_std"]]).all()
    obs, _ = env.reset()
    expected = agent.choose_action(obs, deterministic=True)
    agent.save(tmp_path / "scaled.pt")
    restored, _ = PPOAgent.load(tmp_path / "scaled.pt", "cpu", True)
    actual = restored.choose_action(obs, deterministic=True)
    assert expected[0] == actual[0]
    assert expected[2:] == pytest.approx(actual[2:])


def test_zero_actor_is_uniform_without_handcrafted_prior(tiny_config, tmp_path):
    config = stabilized(tiny_config)
    env = LeoEnv(config, task_trace=((Task(0, 0, 10, 1, 10, 0), Task(1, 0, 30, 1, 10, 0)), (), ()))
    obs, _ = env.reset()
    agent = PPOAgent(config)
    with torch.no_grad():
        agent.model.scorer.score[-1].weight.zero_()
        agent.model.scorer.score[-1].bias.zero_()
    actions, batch, log, _ = agent.choose_action(obs, deterministic=True)
    for decision in batch.decisions:
        logits = agent.model.logits(agent.model.encode(batch.graph), decision)
        assert torch.equal(logits, torch.zeros_like(logits))
    assert all(d.mask[a] for d, a in zip(batch.decisions, batch.actions))
    assert log == pytest.approx(float(agent.model.evaluate_batch(batch)[0].detach()), abs=1e-5)
    assert env.engine.jobs == {}
    rows = run_comparison(config, ["batch_greedy", "node_greedy"], tmp_path / "compare")
    assert len({r["task_trace_sha256"] for r in rows}) == 1
    assert len({r["cpu_capacities_sha256"] for r in rows}) == 1


@pytest.mark.parametrize("key,value", [("value_scale", 0), ("value_scale", float("inf")),
                                       ("completion_prior_strength", -1), ("completion_prior_strength", True)])
def test_invalid_learning_scales_are_rejected(tiny_config, key, value):
    config = stabilized(tiny_config)
    config["rl"][key] = value
    with pytest.raises(ValueError):
        rl_settings(config)


def test_default_scale_and_checkpoint_roundtrip(tiny_config, tmp_path):
    config = deepcopy(tiny_config)
    config["rl"] = {"hidden_dim": 16, "device": "cpu"}
    agent = PPOAgent(config)
    assert agent.settings["value_scale"] == 100 and "completion_prior_strength" not in agent.settings
    env = LeoEnv(config)
    obs, _ = env.reset()
    expected = agent.choose_action(obs, deterministic=True)
    agent.save(tmp_path / "legacy.pt")
    restored, _ = PPOAgent.load(tmp_path / "legacy.pt", "cpu")
    assert restored.choose_action(obs, deterministic=True)[2:] == pytest.approx(expected[2:])
