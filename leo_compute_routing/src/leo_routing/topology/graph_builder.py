import numpy as np

from .walker import EARTH_RADIUS_M


def line_of_sight(first, second, clearance_m=0.0):
    """Check the closest point of the complete line segment against Earth."""
    delta = second - first
    length_squared = float(np.dot(delta, delta))
    if length_squared == 0:
        return False
    fraction = np.clip(-np.dot(first, delta) / length_squared, 0.0, 1.0)
    return np.linalg.norm(first + fraction * delta) > EARTH_RADIUS_M + clearance_m


def build_adjacency(positions, settings):
    count = len(positions)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adjacency = np.zeros((count, count), dtype=bool)
    degrees = np.zeros(count, dtype=int)
    per_plane, planes = settings["sats_per_plane"], settings["planes"]

    def eligible(i, j):
        return (distances[i, j] <= settings["max_isl_distance_m"] and
                line_of_sight(positions[i], positions[j], settings["earth_clearance_m"]))

    def connect(i, j):
        if not adjacency[i, j]:
            adjacency[i, j] = adjacency[j, i] = True
            degrees[i] += 1
            degrees[j] += 1

    for plane in range(planes):
        for member in range(per_plane):
            i, j = plane * per_plane + member, plane * per_plane + (member + 1) % per_plane
            if eligible(i, j):
                connect(i, j)
    latitudes = np.rad2deg(np.arcsin(positions[:, 2] / np.linalg.norm(positions, axis=1)))
    crosslinks = []
    for i in range(count):
        for j in range(i + 1, count):
            separation = abs(i // per_plane - j // per_plane)
            if not (planes > 1 and separation in (1, planes - 1)):
                continue
            if max(abs(latitudes[i]), abs(latitudes[j])) > settings["crosslink_latitude_limit_deg"]:
                continue
            if eligible(i, j):
                crosslinks.append((distances[i, j], i, j))
    for _, i, j in sorted(crosslinks):
        if degrees[i] < settings["max_degree"] and degrees[j] < settings["max_degree"]:
            connect(i, j)
    # Disconnected constellations are allowed; never fabricate links to force connectivity.
    return adjacency, distances


def periodic_adjacency(count, slot, change_slots):
    """Synthetic ring with alternating matching chords, for software stress tests."""
    adjacency = np.zeros((count, count), dtype=bool)
    for i in range(count):
        j = (i + 1) % count
        adjacency[i, j] = adjacency[j, i] = True
    phase = (slot // change_slots) % 2
    for i in range(phase, count - 2, 2):
        j = (i + count // 2) % count
        if i != j:
            adjacency[i, j] = adjacency[j, i] = True
    return adjacency
