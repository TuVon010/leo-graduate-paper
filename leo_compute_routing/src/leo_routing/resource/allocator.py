import math

import numpy as np


def allocate_capacity(workloads, capacity, mode="sqrt", weights=None):
    """Minimize sum w_i * W_i / x_i for a fixed active set (sqrt mode).

    The weights and workloads here are positive constants during a service
    interval. This is not an optimizer for the dynamic global scheduling problem.
    """
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Resource capacity must be positive and finite")
    if mode not in ("sqrt", "equal"):
        raise ValueError("Unknown allocation mode")
    keys = tuple(workloads)
    if not keys:
        return {}
    sizes = np.array([workloads[key] for key in keys], dtype=float)
    importance = np.array([1.0 if weights is None else weights[key] for key in keys], dtype=float)
    if (not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(importance)) or
            np.any(sizes <= 0) or np.any(importance <= 0)):
        raise ValueError("Allocation requires positive finite workloads and weights")
    shares = np.sqrt(sizes) * np.sqrt(importance) if mode == "sqrt" else np.ones(len(keys))
    # Normalize first, reducing overflow risk.
    shares /= shares.max()
    rates = capacity * shares / shares.sum()
    allocation = dict(zip(keys, map(float, rates)))
    if sum(allocation.values()) > capacity * (1 + 1e-12):
        raise RuntimeError("Resource allocation exceeded capacity")
    return allocation
