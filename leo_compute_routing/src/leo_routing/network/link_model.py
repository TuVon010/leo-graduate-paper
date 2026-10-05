import math

LIGHT_SPEED_MPS = 299792458.0


def edge_key(first, second):
    """Both directions share one ISL budget in the first version."""
    return (min(first, second), max(first, second))


def path_edges(path):
    return tuple(edge_key(i, j) for i, j in zip(path, path[1:]))


def transmission_delay(data_bits, rate_bps):
    if not math.isfinite(data_bits) or not math.isfinite(rate_bps) or data_bits < 0 or rate_bps <= 0:
        raise ValueError("Transmission requires nonnegative bits and positive finite rate")
    return data_bits / rate_bps


def link_delay(data_bits, rate_bps, distance_m):
    if not math.isfinite(distance_m) or distance_m < 0:
        raise ValueError("Distance must be finite and nonnegative")
    return transmission_delay(data_bits, rate_bps) + distance_m / LIGHT_SPEED_MPS
