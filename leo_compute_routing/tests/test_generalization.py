from copy import deepcopy
import json
import csv

import pytest

pytest.importorskip("torch")

from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.evaluation.generalization import run_generalization


def test_generalization_preserves_checkpoints_and_reports_heldout_anchor(tiny_config, tmp_path):
    config = deepcopy(tiny_config)
    config["rl"] = {"hidden_dim": 16, "device": "cpu"}
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
        run_generalization([checkpoint], {"same": config}, [2026, 201], tmp_path / "overlap")


def test_checkpoint_keeps_own_router_mode_during_ablation(tiny_config):
    config = deepcopy(tiny_config)
    config["rl"] = {"device": "cpu"}
    config["routing"]["mode"] = "snapshot"
    agent = PPOAgent(config)
    evaluation = deepcopy(config)
    evaluation["routing"]["mode"] = "contact"
    assert agent.configure_environment(evaluation)["routing"]["mode"] == "snapshot"


def test_single_seed_pilot_has_no_confidence_interval(tiny_config, tmp_path):
    config = deepcopy(tiny_config)
    config["rl"] = {"device": "cpu", "hidden_dim": 16}
    checkpoint = tmp_path / "model.pt"
    PPOAgent(config).save(checkpoint)
    rows = run_generalization([checkpoint], {"same": config}, [201], tmp_path / "pilot",
                              bootstrap_samples=20)
    assert len(rows) == 1 and rows[0]["seed"] == 201
    manifest = json.loads((tmp_path / "pilot/in_domain/evaluation_study.json").read_text())
    assert manifest["single_seed_pilot"] and not manifest["confidence_intervals_available"]
    for filename in ("in_domain/aggregate.csv", "cost_shift.csv"):
        with (tmp_path / "pilot" / filename).open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                assert row["ci95_low"] == row["ci95_high"] == ""
