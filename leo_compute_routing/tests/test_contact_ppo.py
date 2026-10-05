from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.agents.features import frozen_array
from leo_routing.env.leo_env import LeoEnv
from leo_routing.tasks.task import Task


def contact_config(tiny_config, encoder="gat"):
    config = deepcopy(tiny_config)
    config["routing"].update(candidate_generation="contact", reference_rate_fraction=1.0, lookahead_slots=5)
    config["rl"] = {"encoder": encoder, "hidden_dim": 16, "device": "cpu", "shield_mode": "contact"}
    return config


@pytest.mark.parametrize("encoder", ["mlp", "gat"])
def test_whole_policy_is_equivariant_under_node_and_candidate_relabeling(tiny_config, encoder):
    config = contact_config(tiny_config, encoder)
    env = LeoEnv(config, task_trace=((Task(0, 0, 10, 1, 10, 0),), (), ()))
    observation, _ = env.reset()
    agent = PPOAgent(config)
    _, batch, log, value = agent.choose_action(observation)
    graph, decision = batch.graph, batch.decisions[0]
    permutation = np.array([3, 0, 5, 2, 1, 4])
    inverse = np.argsort(permutation)
    candidates = np.arange(len(decision.mask))[::-1]
    changed_graph = replace(graph, nodes=frozen_array(graph.nodes[permutation]),
                            edge_index=frozen_array(inverse[graph.edge_index], np.int64))
    changed_decision = replace(decision, source=int(inverse[decision.source]),
        destinations=frozen_array(inverse[decision.destinations[candidates]], np.int64),
        path_pool=frozen_array(decision.path_pool[candidates][:, permutation]),
        candidates=frozen_array(decision.candidates[candidates]), mask=frozen_array(decision.mask[candidates], bool),
        predicted_feasible=frozen_array(decision.predicted_feasible[candidates], bool))
    original = agent.model.encode(graph)
    changed = agent.model.encode(changed_graph)
    assert torch.allclose(original[2], changed[2], atol=1e-6)
    assert torch.allclose(agent.model.logits(changed, changed_decision),
                          agent.model.logits(original, decision)[torch.tensor(candidates.copy())], atol=1e-6)
    assert float(agent.model.evaluate_batch(batch)[0].detach()) == pytest.approx(log, abs=1e-5)


def test_contact_shield_rechecks_contact_after_prefix_reservation(tiny_config, trace_factory):
    config = contact_config(tiny_config)
    trace = trace_factory(satellites=6, slots=24, unavailable_slots=tuple(range(2, 24)))
    jobs = ((Task(0, 0, 150, 1, 10, 0), Task(1, 0, 150, 1, 10, 0)), (), ())
    capacities = [1, 1000, 1000, 1000, 1000, 1000]
    contact = LeoEnv(config, trace, jobs, capacities)
    observation, _ = contact.reset()
    agent = PPOAgent(config)
    actions, batch, *_ = agent.choose_action(observation, deterministic=True)
    assert actions == {0: 1, 1: 0}
    assert batch.decisions[1].fallback and not batch.decisions[1].predicted_feasible[0]
    assert agent.last_decision_metrics["predicted_invalid_selection_count"] == 1  # explicit unsafe local fallback
    static_config = deepcopy(config)
    static_config["rl"]["shield_mode"] = "mask"
    static = LeoEnv(static_config, trace, jobs, capacities)
    obs, _ = static.reset()
    static_agent = PPOAgent(static_config)
    static_actions, *_ = static_agent.choose_action(obs, deterministic=True)
    assert static_actions == {0: 1, 1: 1}
    contact.step(actions)
    static.step(static_actions)
    contact.engine.advance(3)
    static.engine.advance(3)
    assert contact.engine.route_failures == 0 and static.engine.route_failures == 2


@pytest.mark.parametrize("encoder", ["mlp", "gat"])
def test_frozen_weights_and_normalization_support_24_48_72_96(tiny_config, encoder):
    config = contact_config(tiny_config, encoder)
    config["topology"].update(planes=6, sats_per_plane=8)
    config["tasks"]["arrival_rate_per_slot"] = 8
    agent = PPOAgent(config)
    original_weights = {key: value.detach().clone() for key, value in agent.model.state_dict().items()}
    scales = (agent.features.per_node_task_scale, agent.features.max_bits, agent.features.max_complexity)
    for planes in (3, 6, 9, 12):
        target = deepcopy(config)
        target["topology"]["planes"] = planes
        task = Task(0, 0, 10, 1, 10, 0)
        env = LeoEnv(target, task_trace=((task,), (), ()))
        observation, _ = env.reset()
        actions, batch, log, value = agent.choose_action(observation, deterministic=True)
        env.step(actions)
        assert batch.graph.nodes.shape[0] == planes * 8
        assert np.isfinite([log, value]).all()
        assert (agent.features.per_node_task_scale, agent.features.max_bits, agent.features.max_complexity) == scales
    assert all(torch.equal(value, original_weights[key]) for key, value in agent.model.state_dict().items())


def test_schema1_checkpoint_is_rejected_with_retraining_instruction(tiny_config, tmp_path):
    agent = PPOAgent(contact_config(tiny_config))
    checkpoint = tmp_path / "old.pt"
    agent.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload["feature_schema"] = 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Retrain"):
        PPOAgent.load(checkpoint)
