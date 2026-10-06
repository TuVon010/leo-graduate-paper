import json
from copy import deepcopy

import pytest
pytest.importorskip("torch")

from leo_routing.agents.ppo_agent import PPOAgent
from leo_routing.evaluation.rl_evaluator import evaluate_checkpoints


def test_validation_reuse_is_explicit_and_training_overlap_remains_rejected(tiny_config, tmp_path):
    config = deepcopy(tiny_config)
    config["rl"] = {"hidden_dim": 16, "device": "cpu", "train_seeds": [2026], "validation_seeds": [100]}
    checkpoint = tmp_path / "model.pt"
    PPOAgent(config).save(checkpoint)
    with pytest.raises(ValueError, match="overlap validation"):
        evaluate_checkpoints([checkpoint], [100], tmp_path / "strict")
    with pytest.raises(ValueError, match="overlap checkpoint training"):
        evaluate_checkpoints([checkpoint], [2026], tmp_path / "train_overlap", allow_validation_reuse=True)
    messages = []
    rows = evaluate_checkpoints([checkpoint], [100], tmp_path / "reuse", ["local"],
                                progress=messages.append, allow_validation_reuse=True)
    study = json.loads((tmp_path / "reuse/evaluation_study.json").read_text())
    assert study["evaluation_role"] == "validation_reuse"
    assert not study["independent_test"] and not study["held_out_seed_check"]
    assert study["checkpoint_selection_uses_evaluation_seeds"]
    assert study["validation_reuse_seeds_by_policy"] == {"gat_ppo": [100]}
    assert any("validation_reuse" in text for text in messages)
    assert len({row["task_trace_sha256"] for row in rows}) == 1
    assert len({row["cpu_capacities_sha256"] for row in rows}) == 1
    evaluate_checkpoints([checkpoint], [301], tmp_path / "heldout")
    independent = json.loads((tmp_path / "heldout/evaluation_study.json").read_text())
    assert independent["independent_test"] and independent["evaluation_role"] == "held_out_test"
