from dataclasses import dataclass
import math

from .link_model import LIGHT_SPEED_MPS


@dataclass(frozen=True)
class RoutePrediction:
    route_seconds: float
    topology_feasible: bool
    fully_checked: bool
    checked_hops: int
    bottleneck_bps: float


def predict_route(task, path, slot, trace, rate_fraction, lookahead_slots):
    """Prediction only: competing current/future tasks may reduce actual rates.

    A topology snapshot describes [t*dt, (t+1)*dt). Check every snapshot
    touched by transmission, excluding propagation after the final bit has
    left. Missing future coverage is marked unknown, never silently certified.
    H=0 is the explicit snapshot-only ablation.
    """
    if len(path) == 1:
        return RoutePrediction(0.0, True, True, 0, 0.0)
    start = slot * trace.slot_seconds
    elapsed, checked, bottleneck = 0.0, 0, float("inf")
    feasible, fully = True, lookahead_slots > 0
    coverage_end = min(trace.slots, slot + lookahead_slots + 1) * trace.slot_seconds
    for first, second in zip(path, path[1:]):
        capacity = float(trace.capacities[slot, first, second])
        if not trace.adjacency[slot, first, second] or capacity <= 0:
            return RoutePrediction(float("inf"), False, False, checked, 0.0)
        bottleneck = min(bottleneck, capacity)
        # Conservative rate fraction and minimum capacity in checked snapshots.
        rate = capacity * rate_fraction
        hop_start = start + elapsed
        # A decreasing fixed-point sequence of rates; terminate at finite cache minimum.
        while lookahead_slots > 0:
            hop_end = hop_start + task.data_bits / rate
            interval_end = min(hop_end, coverage_end)
            if interval_end <= hop_start:
                break
            first_slot = max(slot, int(math.floor(hop_start / trace.slot_seconds + 1e-10)))
            last_slot = int(math.ceil(interval_end / trace.slot_seconds - 1e-10)) - 1
            if last_slot < first_slot:
                break
            availability = trace.adjacency[first_slot:last_slot + 1, first, second]
            if not availability.all():
                feasible = False
                break
            minimum = float(trace.capacities[first_slot:last_slot + 1, first, second].min())
            updated_rate = min(rate, minimum * rate_fraction)
            if updated_rate >= rate * (1 - 1e-12):
                break
            rate = updated_rate
        hop_end = hop_start + task.data_bits / rate
        if lookahead_slots > 0 and hop_end <= coverage_end + 1e-10:
            checked += 1
        else:
            fully = False
        elapsed += task.data_bits / rate + trace.distances[slot, first, second] / LIGHT_SPEED_MPS
    return RoutePrediction(float(elapsed), feasible, fully, checked, bottleneck)
