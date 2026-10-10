from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.env.leo_env import LeoEnv


def setup(tiny_config, encoder="gated_gat", neighbors="physical"):
    config = deepcopy(tiny_config)
    config["rl"] = dict(encoder=encoder, feature_set="kkt", graph_neighbors=neighbors,
                        hidden_dim=16, device="cpu")
    agent = PPOAgent(config)
    obs, _ = LeoEnv(config).reset()
    _, batch, _, _ = agent.choose_action(obs)
    return agent, batch


def test_self_graph_encoding_ignores_physical_neighbors(tiny_config):
    agent, batch = setup(tiny_config, "gat", "self")
    graph = batch.graph
    changed = replace(graph, edge_index=graph.edge_index[:, :0], edges=graph.edges[:0])
    first = agent.model.encode(graph); second = agent.model.encode(changed)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_gated_critic_has_no_graph_gradient_and_gate_can_learn(tiny_config):
    agent, batch = setup(tiny_config)
    model = agent.model
    model.encode(batch.graph)[2].backward()
    assert all(p.grad is None for p in model.graph_encoder.parameters())
    model.zero_grad(set_to_none=True)
    encoded = model.encode(batch.graph)
    log, _, _ = model.evaluate_batch(batch)
    log.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.graph_encoder.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.scorer.gate.parameters())
    with torch.no_grad():
        # Closing the gate leaves the exact own-node scoring branch.
        model.scorer.gate[-2].weight.zero_(); model.scorer.gate[-2].bias.fill_(-100)
        original = model.logits(encoded, batch.decisions[0])
        changed = model.logits(encoded[:3] + (encoded[3] * 100,), batch.decisions[0])
        assert torch.allclose(original, changed, atol=1e-6)


def test_new_schema_checkpoint_roundtrip(tiny_config, tmp_path):
    agent, batch = setup(tiny_config)
    expected = agent.model.evaluate_batch(batch)
    agent.save(tmp_path / "gated.pt")
    restored, _ = PPOAgent.load(tmp_path / "gated.pt", "cpu", True)
    actual = restored.model.evaluate_batch(batch)
    assert restored.features.schema == 4
    for a, b in zip(expected, actual):
        assert torch.equal(a, b)
    assert restored.name == "gated_gat_ppo_kkt"
