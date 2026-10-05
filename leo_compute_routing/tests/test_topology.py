from copy import deepcopy

import numpy as np
import pytest

from leo_routing.topology.graph_builder import line_of_sight
from leo_routing.topology.topology_cache import TopologyTrace, generate_topology, topology_signature
from leo_routing.topology.walker import EARTH_RADIUS_M, walker_positions


def test_walker_altitude_and_motion(tiny_config):
    settings = tiny_config["topology"]
    positions = walker_positions(settings, [0.0, 100.0])
    assert np.allclose(np.linalg.norm(positions, axis=-1), EARTH_RADIUS_M + settings["altitude_m"])
    assert not np.allclose(positions[0], positions[1])


def test_earth_blockage():
    radius = EARTH_RADIUS_M + 600000.0
    assert not line_of_sight(np.array([radius, 0, 0]), np.array([-radius, 0, 0]))
    assert line_of_sight(np.array([radius, 0, 0]), np.array([radius, 1000, 0]))


def test_cache_round_trip_and_configuration_guard(tiny_config, tmp_path):
    trace = generate_topology(tiny_config)
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = TopologyTrace.load(path, topology_signature(tiny_config))
    assert np.array_equal(loaded.adjacency, trace.adjacency)
    assert np.array_equal(loaded.positions, trace.positions)
    with pytest.raises(ValueError):
        TopologyTrace.load(path, "different")
    assert not loaded.adjacency.flags.writeable


def test_walker_links_satisfy_geometry_and_degree(tiny_config):
    config = deepcopy(tiny_config)
    config["topology"].update(mode="walker", planes=3, sats_per_plane=8)
    trace = generate_topology(config)
    for slot in range(trace.slots):
        graph = trace.graph(slot)
        assert max(dict(graph.degree()).values()) <= config["topology"]["max_degree"]
        for i, j, edge in graph.edges(data=True):
            assert edge["distance_m"] <= config["topology"]["max_isl_distance_m"]
            assert line_of_sight(trace.positions[slot, i], trace.positions[slot, j],
                                 config["topology"]["earth_clearance_m"])


def test_disconnection_is_preserved(tiny_config):
    config = deepcopy(tiny_config)
    config["topology"].update(mode="walker", planes=1, phase_factor=0, max_isl_distance_m=1.0)
    trace = generate_topology(config)
    assert trace.graph(0).number_of_edges() == 0
