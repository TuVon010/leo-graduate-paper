from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-rl.txt to test neural methods")

from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.agents.rollout_buffer import RolloutBuffer, Transition
from leo_routing.agents.trainer import train_ppo
from leo_routing.env.leo_env import LeoEnv
from leo_routing.evaluation.rl_evaluator import evaluate_checkpoints
from leo_routing.models.gat_encoder import AttentionLayer
from leo_routing.tasks.task import Task


def configured(tiny_config, encoder="gat"):
    config = deepcopy(tiny_config)
    config["rl"] = {"encoder": encoder, "hidden_dim": 16, "epochs": 2, "device": "cpu",
                    "updates": 2, "episodes_per_update": 1, "minibatch_steps": 4,
                    "train_seeds": [42, 43], "validation_seeds": [101],
                    "validation_every": 1, "checkpoint_every": 1}
    return config


@pytest.mark.parametrize("encoder", ["mlp", "gat"])
def test_joint_log_probability_and_mask_snapshot(tiny_config, encoder):
    config = configured(tiny_config, encoder)
    agent = PPOAgent(config)
    env = LeoEnv(agent.configure_environment(config))
    observation, _ = env.reset()
    actions, batch, old_log, old_value = agent.choose_action(observation)
    log, entropy, value = agent.model.evaluate_batch(batch)
    assert float(log.detach()) == pytest.approx(old_log, abs=1e-5)
    assert float(value.detach()) == pytest.approx(old_value, abs=1e-6)
    assert torch.isfinite(entropy)
    for decision, action in zip(batch.decisions, batch.actions):
        distribution = agent.model.distribution(agent.model.encode(batch.graph), decision)
        assert decision.mask[action]
        assert torch.all(distribution.probs[~torch.tensor(decision.mask.copy())] == 0)
        assert not decision.mask.flags.writeable
    env.step(actions)
    new_log = agent.model.evaluate_batch(batch)[0]
    assert float(new_log.detach()) == pytest.approx(old_log, abs=1e-5)


def test_all_infeasible_uses_local_fallback(tiny_config):
    config = configured(tiny_config)
    trace = ((Task(0, 0, 1e9, 1e3, 0.001, 0),), (), ())
    env = LeoEnv(config, task_trace=trace)
    obs, _ = env.reset()
    agent = PPOAgent(config)
    actions, batch, log, value = agent.choose_action(obs)
    assert actions[0].is_local and actions[0].compute_sat == 0
    assert batch.decisions[0].fallback
    assert np.count_nonzero(batch.decisions[0].mask) == 1
    assert np.isfinite([log, value]).all()
    assert not batch.has_choice


def test_gat_is_permutation_equivariant_and_respects_non_neighbors():
    torch.manual_seed(3)
    layer = AttentionLayer(16, 4, 4)
    nodes = torch.randn(3, 16, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edges = torch.randn(2, 4, requires_grad=True)
    original = layer(nodes, edge_index, edges)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    relabeled = layer(nodes[permutation], inverse[edge_index], edges)
    assert torch.allclose(relabeled, original[permutation], atol=1e-6)
    changed = nodes.detach().clone()
    changed[2] += torch.arange(16)
    assert torch.allclose(layer(changed, edge_index, edges)[0], original[0], atol=1e-6)
    original[0, 0].backward()
    assert torch.isfinite(nodes.grad).all() and torch.isfinite(edges.grad).all()
    isolated = layer(nodes.detach(), torch.empty(2, 0, dtype=torch.long), torch.empty(0, 4))
    assert torch.isfinite(isolated).all()


@pytest.mark.parametrize("encoder", ["mlp", "gat"])
def test_real_rollout_updates_weights_and_roundtrips_checkpoint(tiny_config, tmp_path, encoder):
    config = configured(tiny_config, encoder)
    agent = PPOAgent(config)
    env = LeoEnv(agent.configure_environment(config))
    obs, _ = env.reset()
    buffer = RolloutBuffer()
    while True:
        actions, batch, log, value = agent.choose_action(obs)
        obs, reward, done, truncated, _ = env.step(actions)
        terminal = done or truncated
        buffer.append(Transition(batch, log, value, reward, 0 if terminal else agent.value(obs), terminal))
        if terminal:
            break
    before = {name: p.detach().clone() for name, p in agent.model.named_parameters()}
    metrics = agent.update(buffer)
    assert metrics["optimizer_steps"] > 0
    assert all(np.isfinite(v) for k, v in metrics.items() if v is not None)
    assert any(not torch.equal(p, before[name]) for name, p in agent.model.named_parameters())
    obs, _ = env.reset()
    expected = agent.choose_action(obs, deterministic=True)
    agent.save(tmp_path / "checkpoint.pt", {"episode_count": 1})
    restored, state = PPOAgent.load(tmp_path / "checkpoint.pt", "cpu", True)
    actual = restored.choose_action(obs, deterministic=True)
    assert expected[0] == actual[0] and actual[2:] == pytest.approx(expected[2:])
    assert state == {"episode_count": 1}
    assert restored.optimizer.state_dict()["state"]


def test_empty_slots_train_critic_without_actor_updates(tiny_config):
    config = configured(tiny_config)
    config["tasks"]["arrival_rate_per_slot"] = 0
    agent = PPOAgent(config)
    env = LeoEnv(config)
    obs, _ = env.reset()
    actions, batch, log, value = agent.choose_action(obs)
    assert actions == {} and log == 0 and not batch.has_choice
    parameters = {name: p.detach().clone() for name, p in agent.model.scorer.named_parameters()}
    buffer = RolloutBuffer()
    buffer.append(Transition(batch, log, value, 1, 0, True))
    metrics = agent.update(buffer)
    assert metrics["policy_steps"] == 0 and metrics["value_loss"] > 0
    assert all(torch.equal(p, parameters[name]) for name, p in agent.model.scorer.named_parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_cuda_policy_can_be_loaded_on_cpu(tiny_config, tmp_path):
    config = configured(tiny_config)
    agent = PPOAgent(config, "cuda")
    env = LeoEnv(config)
    obs, _ = env.reset()
    actions, batch, log, value = agent.choose_action(obs, deterministic=True)
    assert np.isfinite([log, value]).all()
    evaluated = agent.model.evaluate_batch(batch)
    evaluated[2].square().backward()
    assert any(p.grad is not None for p in agent.model.graph_encoder.parameters())
    agent.save(tmp_path / "cuda.pt")
    cpu, _ = PPOAgent.load(tmp_path / "cuda.pt", "cpu")
    restored = cpu.choose_action(obs, deterministic=True)
    assert restored[0] == actions
    assert restored[2:] == pytest.approx([log, value], abs=1e-4)


def test_training_resume_and_independent_evaluation(tiny_config, tmp_path):
    config = configured(tiny_config, "mlp")
    train_ppo(config, tmp_path / "train")
    assert (tmp_path / "train/best.pt").exists()
    assert (tmp_path / "train/updates.csv").exists()
    config["rl"]["updates"] = 3
    resumed = train_ppo(config, tmp_path / "resume", tmp_path / "train/last.pt")
    assert resumed.update_count == 3
    uninterrupted = train_ppo(config, tmp_path / "uninterrupted")
    assert all(torch.equal(a, b) for a, b in zip(resumed.model.parameters(), uninterrupted.model.parameters()))
    with pytest.raises(ValueError, match="overlap"):
        evaluate_checkpoints([tmp_path / "train/best.pt"], [42, 201], tmp_path / "bad_test")
    rows = evaluate_checkpoints([tmp_path / "train/best.pt"], [201, 202], tmp_path / "evaluation",
                                ["local"], bootstrap_samples=100)
    for seed in [201, 202]:
        same_seed = [row for row in rows if row["seed"] == seed]
        assert len({row["task_trace_sha256"] for row in same_seed}) == 1
        assert len({row["cpu_capacities_sha256"] for row in same_seed}) == 1
    assert (tmp_path / "evaluation/aggregate.csv").exists()


def test_cpu_auto_fallback_and_training_metrics_are_reported(tiny_config, tmp_path, monkeypatch):
    import csv
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = configured(tiny_config, "mlp")
    config["rl"].update(device="auto", updates=1, episodes_per_update=1)
    messages = []
    agent = train_ppo(config, tmp_path / "cpu", progress=messages.append, log_interval_seconds=0.001)
    assert agent.device.type == "cpu"
    output = "\n".join(messages)
    assert "CUDA_available=no GPU_used=no" in output
    for text in ("[EPISODE]", "P95=", "task_cost=", "link_util=", "[PPO]", "KL=", "[VALIDATION MEAN]",
                 "[UPDATE DONE] 1/1 (100.0%)", "FPS=", "result_dir="):
        assert text in output
    with (tmp_path / "cpu/updates.csv").open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["rollout_fps"]) > 0
    assert int(row["session_environment_steps"]) == int(row["rollout_steps"])
    assert float(row["eta_seconds"]) == 0
