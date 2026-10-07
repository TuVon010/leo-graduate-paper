from copy import deepcopy

import pytest

from conftest import ROOT
from leo_routing.config import load_config, validate_config


def test_inheritance_and_override():
    config = load_config(ROOT / "configs/smoke.yaml", ["tasks.arrival_rate_per_slot=0", "routing.lookahead_slots=0"])
    assert config["topology"]["mode"] == "periodic"
    assert config["topology"]["altitude_m"] == 600000
    assert config["tasks"]["arrival_rate_per_slot"] == 0


@pytest.mark.parametrize("key,value", [("slots", 0), ("slot_seconds", 0), ("slots", 3.5), ("drop_at_deadline", "false")])
def test_invalid_simulation_rejected(tiny_config, key, value):
    config = deepcopy(tiny_config)
    config["simulation"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_unknown_override_and_cycle_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(ROOT / "configs/base.yaml", ["routing.nonexistent=2"])
    (tmp_path / "a.yaml").write_text("extends: b.yaml", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("key", ["k_paths", "candidate_generation", "path_backend", "reference_data_bits"])
def test_retired_routing_parameters_are_rejected(tiny_config, key):
    tiny_config["routing"][key] = 1
    with pytest.raises(ValueError, match="retired"):
        validate_config(tiny_config)
