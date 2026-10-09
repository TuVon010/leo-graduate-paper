"""Bounded contact windows and no-disruption store-and-forward predictions.

Only the configured rolling horizon is visible. Busy-link waiting is allowed;
waiting for a later disconnected contact is deliberately not implemented.
Reservations describe a prediction calendar, not enforced service guarantees.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from .link_model import edge_key, LIGHT_SPEED_MPS

EPS = 1e-10


@dataclass(frozen=True)
class ContactWindow:
    edge: tuple
    start: float
    end: float
    capacity_bits: float
    end_known: bool


@dataclass(frozen=True)
class HopPrediction:
    edge: tuple
    start: float
    end: float
    arrival: float
    segments: tuple
    feasible: bool
    fully_checked: bool
    contact_margin_seconds: float
    capacity_margin_bits: float


@dataclass(frozen=True)
class ContactRoutePrediction:
    route_seconds: float
    topology_feasible: bool
    fully_checked: bool
    bottleneck_bps: float
    contact_margin_seconds: float
    capacity_margin_ratio: float
    hops: tuple


@dataclass(frozen=True, eq=False)
class ContactPlan:
    start: float
    end: float
    slot_seconds: float
    adjacency: np.ndarray
    capacities: np.ndarray
    distances: np.ndarray
    future_enabled: bool
    windows: tuple
    windows_by_edge: object

    @classmethod
    def from_trace(cls, trace, slot, lookahead_slots):
        stop = min(trace.slots, slot + lookahead_slots + 1)
        # TopologyTrace owns immutable arrays; read-only views preserve snapshots.
        arrays = [a[slot:stop] for a in
                  (trace.adjacency, trace.capacities, trace.distances)]
        for array in arrays:
            array.setflags(write=False)
        adjacency, capacities, distances = arrays
        start, end = slot * trace.slot_seconds, stop * trace.slot_seconds
        windows = []
        stable = np.all(adjacency == adjacency[0])
        for i, j in zip(*np.where(np.triu(adjacency.any(axis=0), 1))):
            if stable:
                windows.append(ContactWindow((int(i), int(j)), start, end,
                    float(capacities[:, i, j].sum() * trace.slot_seconds), False))
                continue
            active = adjacency[:, i, j]
            changes = np.diff(np.r_[False, active, False].astype(int))
            for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                windows.append(ContactWindow((int(i), int(j)), start + first * trace.slot_seconds,
                    start + last * trace.slot_seconds,
                    float(capacities[first:last, i, j].sum() * trace.slot_seconds), last < len(active)))
        grouped = {}
        for window in windows:
            grouped.setdefault(window.edge, []).append(window)
        grouped = MappingProxyType({edge: tuple(items) for edge, items in grouped.items()})
        return cls(start, end, trace.slot_seconds, *arrays, lookahead_slots > 0, tuple(windows), grouped)

    def _index(self, seconds):
        return max(0, min(len(self.adjacency) - 1,
                          int(math.floor((seconds - self.start) / self.slot_seconds + EPS))))

    def _window(self, edge, seconds):
        return next((w for w in self.windows_by_edge.get(edge, ()) if
                     w.start <= seconds + EPS and seconds < w.end - EPS), None)

    def transmit(self, first, second, seconds, bits, rate_fraction, reservations=()):
        edge = edge_key(first, second)
        index = self._index(seconds)
        capacity = float(self.capacities[index, first, second])
        # An uncovered tail uses only the LAST VISIBLE rate, and is marked unknown.
        window = self._window(edge, seconds)
        outside = seconds >= self.end - EPS
        possible = bool(self.adjacency[index, first, second]) and (window is not None or outside)
        if capacity <= 0 or not possible:
            fallback_rate = max(float(self.capacities[:, first, second].max()), 1.0) * rate_fraction
            finish = seconds + bits / fallback_rate
            return HopPrediction(edge, seconds, finish, finish, (), False, self.future_enabled and not outside, 0.0, -bits)
        limit = window.end if window else self.end
        end_known = bool(window and window.end_known)
        cursor, remaining, segments = seconds, bits, []
        reserved = sorted(reservations)
        # Raw free service volume within this contact, used as a feature only.
        free_bits = 0.0
        for k in range(len(self.adjacency)):
            left = max(seconds, self.start + k * self.slot_seconds)
            right = min(limit, self.start + (k + 1) * self.slot_seconds)
            if right <= left:
                continue
            occupied = sum(max(0.0, min(right, b) - max(left, a)) for a, b in reserved)
            free_bits += max(0.0, right - left - occupied) * self.capacities[k, first, second] * rate_fraction
        while remaining > max(1e-9, bits * 1e-12):
            if end_known and cursor >= limit - EPS:
                finish = cursor + remaining / (capacity * rate_fraction)
                return HopPrediction(edge, seconds, finish, finish, tuple(segments), False, True,
                                     limit - finish, free_bits - bits)
            if cursor >= self.end - EPS:
                k, boundary = len(self.adjacency) - 1, float("inf")
            else:
                k = self._index(cursor)
                boundary = min(limit, self.start + (k + 1) * self.slot_seconds)
            blocker = next(((a, b) for a, b in reserved if a <= cursor + EPS and b > cursor + EPS), None)
            if blocker:
                cursor = min(blocker[1], limit) if end_known else blocker[1]
                continue
            next_booking = min((a for a, b in reserved if a > cursor + EPS), default=float("inf"))
            boundary = min(boundary, next_booking)
            rate = float(self.capacities[k, first, second]) * rate_fraction
            if rate <= 0 or not self.adjacency[k, first, second]:
                finish = cursor + remaining / (capacity * rate_fraction)
                return HopPrediction(edge, seconds, finish, finish, tuple(segments), False, True,
                                     limit - finish, free_bits - bits)
            duration = min(remaining / rate, max(0.0, boundary - cursor))
            if duration <= EPS and math.isfinite(boundary):
                cursor = boundary
                continue
            segments.append((cursor, cursor + duration))
            cursor += duration
            remaining -= rate * duration
        propagation = float(self.distances[self._index(seconds), first, second]) / LIGHT_SPEED_MPS
        checked = self.future_enabled and cursor <= self.end + EPS
        return HopPrediction(edge, seconds, cursor, cursor + propagation, tuple(segments), True, checked,
                             limit - cursor, free_bits - bits)

    def predict(self, task, path, rate_fraction, links=None, start=None, first_bits=None):
        arrival = self.start if start is None else start
        origin, hops, bottleneck = arrival, [], float("inf")
        for index, (first, second) in enumerate(zip(path, path[1:])):
            bits = first_bits if index == 0 and first_bits is not None else task.data_bits
            hop = self.transmit(first, second, arrival, bits, rate_fraction,
                                (links or {}).get(edge_key(first, second), ()))
            hops.append(hop)
            bottleneck = min(bottleneck, float(self.capacities[self._index(arrival), first, second]))
            arrival = hop.arrival
        return ContactRoutePrediction(arrival - origin, all(h.feasible for h in hops),
            all(h.fully_checked for h in hops), 0.0 if not hops else bottleneck,
            min((h.contact_margin_seconds for h in hops), default=0.0),
            min((h.capacity_margin_bits / task.data_bits for h in hops), default=0.0), tuple(hops))

    def append_hop(self, task, prefix, first, second, rate_fraction, links=None):
        """Reuse a search label's prefix instead of reintegrating previous hops."""
        start = self.start + prefix.route_seconds
        hop = self.transmit(first, second, start, task.data_bits, rate_fraction,
                            (links or {}).get(edge_key(first, second), ()))
        capacity = float(self.capacities[self._index(start), first, second])
        return ContactRoutePrediction(hop.arrival - self.start, prefix.topology_feasible and hop.feasible,
            prefix.fully_checked and hop.fully_checked,
            min(prefix.bottleneck_bps, capacity) if prefix.hops else capacity,
            min(prefix.contact_margin_seconds, hop.contact_margin_seconds) if prefix.hops else hop.contact_margin_seconds,
            min(prefix.capacity_margin_ratio, hop.capacity_margin_bits / task.data_bits) if prefix.hops else
                hop.capacity_margin_bits / task.data_bits, prefix.hops + (hop,))
