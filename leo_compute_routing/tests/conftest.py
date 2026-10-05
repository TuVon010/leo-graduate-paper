from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_routing.config import load_config
from leo_routing.topology.topology_cache import TopologyTrace


@pytest.fixture
def trace_factory():
    def make(satellites=2, slots=8, slot_seconds=1.0, edges=((0, 1),), capacity=100.0,
             propagation_seconds=0.0, unavailable_slots=()):
        adjacency = np.zeros((slots, satellites, satellites), dtype=bool)
        distances = np.zeros_like(adjacency, dtype=float)
        for i, j in edges:
            adjacency[:, i, j] = adjacency[:, j, i] = True
            distances[:, i, j] = distances[:, j, i] = propagation_seconds * 299792458.0
        for slot in unavailable_slots:
            adjacency[slot] = False
        return TopologyTrace(np.zeros((slots, satellites, 3)), adjacency, distances,
                             adjacency.astype(float) * capacity, slot_seconds)
    return make


@pytest.fixture
def tiny_config():
    config = load_config(ROOT / "configs/smoke.yaml")
    config["simulation"].update(slots=3, drain_slots=20, slot_seconds=1.0)
    config["tasks"].update(arrival_rate_per_slot=2.0, data_bits=[10.0, 20.0],
                           cycles_per_bit=[1.0, 2.0], deadline_seconds=[1.0, 3.0])
    config["compute"]["cpu_cycles_per_second"] = [100.0, 200.0]
    config["topology"]["link_capacity_bps"] = 100.0
    return config
