from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from leo_routing.agents.features import frozen_array
from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.env.leo_env import LeoEnv
from leo_routing.tasks.task import Task


@pytest.mark.parametrize("encoder", ["gat", "mlp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_joint_likelihood_entropy_value_and_gradients(tiny_config, encoder, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = deepcopy(tiny_config)
    config["rl"] = dict(encoder=encoder, hidden_dim=16, device=device)
    trace = ((Task(0, 0, 10, 1, 10, 0), Task(1, 2, 20, 1, 10, 0)),
             (Task(2, 1, 10, 1, 10, 1),), ())
    env, agent = LeoEnv(config, task_trace=trace), PPOAgent(config)
    obs, _ = env.reset()
    batches = []
    for _ in range(3):
        actions, batch, _, _ = agent.choose_action(obs)
        batches.append(batch)
        obs, *_ = env.step(actions)
    # Cover variable lengths, reordered slots, single-action masks and empty slots.
    decision = batches[1].decisions[0]
    mask = np.zeros_like(decision.mask)
    mask[batches[1].actions[0]] = True
    batches[1] = replace(batches[1], decisions=(replace(decision, mask=frozen_array(mask, bool)),))
    packed = agent.model.prepare_rollout(batches)
    order = np.array([2, 0, 1])
    scalar = [agent.model.evaluate_batch(batches[i]) for i in order]
    scalar_outputs = tuple(torch.stack([result[j] for result in scalar]) for j in range(3))
    scalar_loss = scalar_outputs[0].sum() + scalar_outputs[1].sum() * .1 + scalar_outputs[2].square().mean()
    scalar_loss.backward()
    gradients = {n: p.grad.clone() for n, p in agent.model.named_parameters() if p.grad is not None}
    agent.model.zero_grad(set_to_none=True)
    batched = agent.model.evaluate_minibatch(packed, order)
    for old, new in zip(scalar_outputs, batched):
        assert torch.allclose(old, new, atol=2e-5, rtol=2e-5)
    loss = batched[0].sum() + batched[1].sum() * .1 + batched[2].square().mean()
    loss.backward()
    for name, parameter in agent.model.named_parameters():
        if name in gradients:
            assert torch.allclose(parameter.grad, gradients[name], atol=3e-4, rtol=3e-4), name
    assert float(batched[0][0].detach()) == 0  # task-free slot


def test_empty_rollout_and_invalid_masks_are_rejected(tiny_config):
    config = deepcopy(tiny_config)
    config["rl"] = dict(hidden_dim=16, device="cpu")
    agent, env = PPOAgent(config), LeoEnv(config)
    obs, _ = env.reset()
    _, batch, _, _ = agent.choose_action(obs)
    with pytest.raises(ValueError, match="empty"):
        agent.model.prepare_rollout([])
    d = replace(batch.decisions[0], mask=frozen_array(np.zeros_like(batch.decisions[0].mask), bool))
    invalid = replace(batch, decisions=(d,), actions=(0,))
    with pytest.raises(ValueError, match="All-masked"):
        agent.model.prepare_rollout([invalid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("encoder", ["gat", "mlp"])
def test_cpu_rollout_matches_cuda_likelihood_and_syncs_after_updates(tiny_config, tmp_path, encoder):
    from leo_routing.agents.rollout_buffer import RolloutBuffer, Transition
    config = deepcopy(tiny_config)
    config["rl"] = dict(encoder=encoder, hidden_dim=16, device="cuda", rollout_device="cpu",
                        epochs=1, minibatch_steps=4)
    agent, env = PPOAgent(config), LeoEnv(config)
    assert agent.model.device.type == "cuda" and agent.rollout_model.device.type == "cpu"
    obs, _ = env.reset()
    buffer = RolloutBuffer()
    while True:
        actions, batch, log, value = agent.choose_action(obs)
        packed = agent.model.prepare_rollout([batch])
        cuda_log, _, cuda_value = agent.model.evaluate_minibatch(packed, [0])
        assert log == pytest.approx(float(cuda_log[0].detach()), abs=2e-5)
        assert value == pytest.approx(float(cuda_value[0].detach()), abs=2e-5)
        obs, reward, done, truncated, _ = env.step(actions)
        terminal = done or truncated
        buffer.append(Transition(batch, log, value, reward, 0 if terminal else agent.value(obs), terminal))
        if terminal:
            break
    agent.update(buffer)
    for cpu, gpu in zip(agent.rollout_model.parameters(), agent.model.parameters()):
        assert torch.equal(cpu, gpu.detach().cpu())
    obs, _ = env.reset()
    expected = agent.choose_action(obs, deterministic=True)
    agent.save(tmp_path / 'split.pt')
    restored, _ = PPOAgent.load(tmp_path / 'split.pt', 'cuda', True)
    actual = restored.choose_action(obs, deterministic=True)
    assert expected[0] == actual[0] and expected[2:] == pytest.approx(actual[2:])
    cpu_agent, _ = PPOAgent.load(tmp_path / 'split.pt', 'cpu')
    assert cpu_agent.rollout_model is cpu_agent.model
    assert cpu_agent.choose_action(obs, deterministic=True)[0] == expected[0]
