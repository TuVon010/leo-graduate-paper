import numpy as np

EARTH_RADIUS_M = 6371000.0
EARTH_MU = 3.986004418e14


def walker_positions(settings, times):
    """Circular two-body Walker-Delta in ECI; not a TLE/SGP4 propagator."""
    planes, per_plane = settings["planes"], settings["sats_per_plane"]
    count = planes * per_plane
    plane = np.repeat(np.arange(planes), per_plane)
    member = np.tile(np.arange(per_plane), planes)
    raan = 2 * np.pi * plane / planes
    phase = 2 * np.pi * member / per_plane + 2 * np.pi * settings["phase_factor"] * plane / count
    radius = EARTH_RADIUS_M + settings["altitude_m"]
    omega = np.sqrt(EARTH_MU / radius ** 3)
    anomaly = np.asarray(times)[:, None] * omega + phase[None, :]
    inclination = np.deg2rad(settings["inclination_deg"])
    ca, sa, cr, sr = np.cos(anomaly), np.sin(anomaly), np.cos(raan), np.sin(raan)
    return radius * np.stack((cr * ca - sr * sa * np.cos(inclination),
                              sr * ca + cr * sa * np.cos(inclination),
                              sa * np.sin(inclination)), axis=-1)
