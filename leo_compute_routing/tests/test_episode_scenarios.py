from copy import deepcopy
import json

import numpy as np
import pytest

from leo_routing.agents.episode_scenarios import episode_configuration, episode_inputs
from leo_routing.agents.settings import rl_settings
from leo_routing.config import load_config, fingerprint
from leo_routing.topology.walker import EARTH_RADIUS_M, EARTH_MU, walker_positions


def randomized(tiny_config):
    config = deepcopy(tiny_config)
    config["topology"].update(mode="walker", planes=3, sats_per_plane=8)
    config["rl"] = dict(randomize_episodes=True, seed=2026, train_seeds=[2026], validation_seeds=[100])
    return config


def test_episode_inputs_change_but_reproduce_across_encoders_and_call_order(tiny_config):
    config = randomized(tiny_config)
    before = fingerprint(config)
    settings = rl_settings(config)
    first, provenance = episode_configuration(config, settings, 0)
    second, other = episode_configuration(config, settings, 1)
    _, first_cpu, first_hash = episode_inputs(first)
    _, second_cpu, second_hash = episode_inputs(second)
    assert first_hash["task_trace_sha256"] != second_hash["task_trace_sha256"]
    assert not np.array_equal(first_cpu, second_cpu)
    assert provenance["episode_seed"] != other["episode_seed"]
    assert provenance["epoch_offset_seconds"] != other["epoch_offset_seconds"]
    assert provenance["hotspot_satellites"] != other["hotspot_satellites"]
    assert len(set(first["tasks"]["hotspot_satellites"])) == len(config["tasks"]["hotspot_satellites"])
    assert first["simulation"]["seed"] not in {2026, 100}
    # Neither encoder choice, policy/global RNG use nor out-of-order requests affects inputs.
    config["rl"]["encoder"] = "mlp"
    np.random.seed(87)
    np.random.random(100)
    episode_configuration(config, rl_settings(config), 98)
    again, repeated = episode_configuration(config, rl_settings(config), 0)
    assert provenance == repeated
    assert episode_inputs(again)[2] == first_hash
    assert np.array_equal(episode_inputs(again)[1], first_cpu)
    config["rl"].pop("encoder")
    assert fingerprint(config) == before


def test_randomized_epoch_preserves_walker_geometry_and_time_shift(tiny_config):
    config = randomized(tiny_config)
    scene, metadata = episode_configuration(config, rl_settings(config), 0)
    offset = metadata["epoch_offset_seconds"]
    radius = EARTH_RADIUS_M + config["topology"]["altitude_m"]
    period = 2 * np.pi * np.sqrt(radius ** 3 / EARTH_MU)
    assert 0 <= offset < period
    positions = walker_positions(scene["topology"], [0, 10])
    expected = walker_positions(config["topology"], [offset, offset + 10])
    assert np.allclose(positions, expected)
    assert np.allclose(np.linalg.norm(positions, axis=-1), radius)
    assert not np.allclose(positions[0], walker_positions(config["topology"], [0])[0])


def test_legacy_replay_and_invalid_episode_indices(tiny_config):
    settings = rl_settings(tiny_config)
    assert settings["randomize_episodes"] is False
    first, _ = episode_configuration(tiny_config, settings, 0)
    second, _ = episode_configuration(tiny_config, settings, 1)
    assert first == second
    for invalid in (-1, True, 1.2):
        with pytest.raises(ValueError):
            episode_configuration(tiny_config, settings, invalid)
    for offset in (-1, float("nan"), True):
        from leo_routing.config import validate_config
        config = deepcopy(tiny_config)
        config["topology"]["epoch_offset_seconds"] = offset
        with pytest.raises(ValueError):
            validate_config(config)


def test_randomized_training_resume_matches_uninterrupted_and_fixed_validation(tiny_config, tmp_path):
    torch = pytest.importorskip("torch")
    from leo_routing.agents.trainer import train_ppo
    import csv
    config = randomized(tiny_config)
    config["rl"].update(encoder="mlp", hidden_dim=16, epochs=1, device="cpu", updates=1,
                        episodes_per_update=1, minibatch_steps=4, validation_every=1, checkpoint_every=1)
    original = train_ppo(config, tmp_path / "first")
    config["rl"]["updates"] = 2
    resumed = train_ppo(config, tmp_path / "resumed", tmp_path / "first/last.pt")
    full = train_ppo(config, tmp_path / "full")
    assert all(torch.equal(a, b) for a, b in zip(resumed.model.parameters(), full.model.parameters()))
    def scenario(directory, episode):
        path = tmp_path / directory / 'episode_scenarios' / ('episode_%06d.json' % episode)
        return json.loads(path.read_text())
    first_scene, uninterrupted_scene = scenario('first', 1), scenario('full', 1)
    for row in (first_scene, uninterrupted_scene):
        row.pop('scenario_preparation_seconds')
        row['resolved_config']['rl'].pop('updates')
    assert first_scene == uninterrupted_scene
    resumed_scene, full_scene = scenario('resumed', 2), scenario('full', 2)
    for row in (resumed_scene, full_scene):
        row.pop('scenario_preparation_seconds')
    assert resumed_scene == full_scene
    assert scenario('full', 1)['task_trace_sha256'] != scenario('full', 2)['task_trace_sha256']
    with (tmp_path / 'full/validation.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r['seed'] for r in rows] == ['100', '100']
    assert len({r['task_count'] for r in rows}) == 1
    assert original.settings['train_seeds'] == [2026]


def test_current_pilot_enables_randomization():
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/experiments/coupled24.yaml')
    assert rl_settings(config)['randomize_episodes'] is True
