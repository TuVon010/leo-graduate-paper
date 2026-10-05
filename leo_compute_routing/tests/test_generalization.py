from copy import deepcopy
import json

import pytest

pytest.importorskip("torch")

from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.evaluation.generalization import run_generalization


def test_generalization_preserves_checkpoints_and_reports_heldout_anchor(tiny_config, tmp_path):
    config = deepcopy(tiny_config)
    config["routing"]["candidate_generation"] = "contact"
    config["rl"] = {"hidden_dim": 16, "device": "cpu", "shield_mode": "contact"}
    agent = PPOAgent(config)
    checkpoint = tmp_path / "model.pt"
    agent.save(checkpoint)
    before = checkpoint.read_bytes()
    larger = deepcopy(config)
    larger["topology"].update(planes=3, sats_per_plane=3)
    larger["tasks"]["arrival_rate_per_slot"] *= 1.5
    rows = run_generalization([checkpoint], {"same": config, "larger": larger}, [201, 202],
                              tmp_path / "study", ["local"], bootstrap_samples=20)
    assert checkpoint.read_bytes() == before
    assert {r["satellites"] for r in rows} == {6, 9}
    assert all(r["relative_cost_shift"] == 0 for r in rows if r["target"] == "same")
    assert (tmp_path / "study/cost_shift.csv").exists()
    assert not (tmp_path / "study/same").exists()  # anchor replay is reused, not simulated twice
    study = json.loads((tmp_path / "study/study.json").read_text())
    assert study["weights_frozen"] and study["feature_scales_from_training_checkpoint"]
    with pytest.raises(ValueError, match="overlap"):
        run_generalization([checkpoint], {"same": config}, [0, 201], tmp_path / "overlap")


def test_checkpoint_keeps_own_candidate_generator_during_ablation(tiny_config):
    config = deepcopy(tiny_config)
    config["rl"] = {"device": "cpu"}
    agent = PPOAgent(config)
    evaluation = deepcopy(config)
    evaluation["routing"]["candidate_generation"] = "contact"
    assert agent.configure_environment(evaluation)["routing"]["candidate_generation"] == "ksp"
