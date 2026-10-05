from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import networkx as nx
import numpy as np

from ..config import fingerprint
from .graph_builder import build_adjacency, periodic_adjacency
from .walker import walker_positions

LIGHT_SPEED_MPS = 299792458.0


@dataclass(eq=False)
class TopologyTrace:
    positions: np.ndarray
    adjacency: np.ndarray
    distances: np.ndarray
    capacities: np.ndarray
    slot_seconds: float
    signature: str = "custom"

    def __post_init__(self):
        count = self.positions.shape[1]
        shape = (len(self.positions), count, count)
        if self.positions.shape != (len(self.positions), count, 3):
            raise ValueError("Invalid position array")
        if any(array.shape != shape for array in (self.adjacency, self.distances, self.capacities)):
            raise ValueError("Inconsistent topology array shapes")
        if (not np.isfinite(self.slot_seconds) or self.slot_seconds <= 0 or len(self.positions) == 0 or
                not all(np.all(np.isfinite(a)) for a in (self.positions, self.distances, self.capacities))):
            raise ValueError("Invalid topology values")
        if not np.array_equal(self.adjacency, self.adjacency.transpose(0, 2, 1)):
            raise ValueError("Topology must use symmetric shared-capacity links")
        if np.any(np.diagonal(self.adjacency, axis1=1, axis2=2)):
            raise ValueError("Self links are forbidden")
        if (np.any(self.distances < 0) or np.any(self.capacities < 0) or
                np.any(self.capacities[self.adjacency] <= 0)):
            raise ValueError("Active links require positive capacity and nonnegative distance")
        if (not np.allclose(self.distances, self.distances.transpose(0, 2, 1)) or
                not np.allclose(self.capacities, self.capacities.transpose(0, 2, 1))):
            raise ValueError("Distances and capacities must be symmetric")
        for array in (self.positions, self.adjacency, self.distances, self.capacities):
            array.setflags(write=False)

    @property
    def satellite_count(self):
        return self.positions.shape[1]

    @property
    def slots(self):
        return len(self.positions)

    def slot_at(self, seconds):
        slot = int(np.floor(seconds / self.slot_seconds + 1e-10))
        if slot < 0 or slot >= self.slots:
            raise IndexError("Time outside topology trace: %s" % seconds)
        return slot

    @lru_cache(maxsize=32)
    def graph(self, slot):
        if slot < 0 or slot >= self.slots:
            raise IndexError("Slot outside topology trace")
        graph = nx.Graph()
        graph.add_nodes_from(range(self.satellite_count))
        for i, j in zip(*np.where(np.triu(self.adjacency[slot], 1))):
            graph.add_edge(int(i), int(j), distance_m=float(self.distances[slot, i, j]),
                           capacity_bps=float(self.capacities[slot, i, j]),
                           propagation_seconds=float(self.distances[slot, i, j] / LIGHT_SPEED_MPS))
        return nx.freeze(graph)

    def diagnostics(self, slots=None):
        length = self.slots if slots is None else min(slots, self.slots)
        edges = [self.graph(t).number_of_edges() for t in range(length)]
        connected = [nx.is_connected(self.graph(t)) for t in range(length)]
        changes = sum(np.count_nonzero(np.triu(self.adjacency[t] != self.adjacency[t - 1], 1))
                      for t in range(1, length))
        return {"satellites": self.satellite_count, "slots": length,
                "mean_edges": float(np.mean(edges)), "connected_slot_fraction": float(np.mean(connected)),
                "edge_state_changes": int(changes), "signature": self.signature}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, positions=self.positions, adjacency=self.adjacency,
                                distances=self.distances, capacities=self.capacities,
                                slot_seconds=np.array(self.slot_seconds), signature=np.array(self.signature))

    @classmethod
    def load(cls, path, expected_signature=None):
        with np.load(path, allow_pickle=False) as cache:
            trace = cls(cache["positions"], cache["adjacency"], cache["distances"], cache["capacities"],
                        float(cache["slot_seconds"]), str(cache["signature"]))
        if expected_signature is not None and trace.signature != expected_signature:
            raise ValueError("Topology cache does not match the experiment configuration")
        return trace


def topology_signature(config):
    sim = config["simulation"]
    return fingerprint({"schema": 1, "topology": config["topology"],
                        "slot_seconds": sim["slot_seconds"],
                        "slots": sim["slots"] + sim["drain_slots"] + 1})


def generate_topology(config):
    sim, topo = config["simulation"], config["topology"]
    length = sim["slots"] + sim["drain_slots"] + 1
    times = np.arange(length) * sim["slot_seconds"]
    positions = walker_positions(topo, times)
    count = positions.shape[1]
    adjacency = np.empty((length, count, count), dtype=bool)
    distances = np.empty((length, count, count), dtype=np.float64)
    for slot in range(length):
        if topo["mode"] == "walker":
            adjacency[slot], distances[slot] = build_adjacency(positions[slot], topo)
        else:
            adjacency[slot] = periodic_adjacency(count, slot, topo["periodic_change_slots"])
            distances[slot] = 1000000.0
            np.fill_diagonal(distances[slot], 0.0)
    capacities = adjacency.astype(float) * topo["link_capacity_bps"]
    return TopologyTrace(positions, adjacency, distances, capacities,
                         sim["slot_seconds"], topology_signature(config))


def get_topology(config, cache_path=None):
    signature = topology_signature(config)
    if cache_path and Path(cache_path).exists():
        return TopologyTrace.load(cache_path, signature)
    trace = generate_topology(config)
    if cache_path:
        trace.save(cache_path)
    return trace
