"""Validated YAML configuration with relative inheritance and CLI overrides."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math

import yaml


def deep_merge(base, patch):
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path, overrides=None, _parents=()):
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("Configuration inheritance cycle: %s" % path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    parent = config.pop("extends", None)
    if parent:
        config = deep_merge(load_config(path.parent / parent, _parents=_parents + (path,)), config)
    for override in overrides or ():
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Override must have the form section.key=value")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError("Unknown configuration key: %s" % key)
            target = target[part]
        if parts[-1] not in target:
            raise ValueError("Unknown configuration key: %s" % key)
        target[parts[-1]] = yaml.safe_load(value)
    validate_config(config)
    return config


def validate_config(config):
    for section in ("simulation", "topology", "tasks", "compute", "resource", "routing", "reward"):
        if section not in config:
            raise ValueError("Missing configuration section: %s" % section)
    sim, topo, tasks, routing = (config[k] for k in ("simulation", "topology", "tasks", "routing"))

    def number(value, name, minimum=0.0, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("%s must be a finite number" % name)
        if value < minimum or (strict and value == minimum):
            raise ValueError("%s is outside its allowed range" % name)

    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("%s must be an integer >= %s" % (name, minimum))

    integer(sim["seed"], "seed", 0)
    integer(sim["slots"], "slots")
    integer(sim["drain_slots"], "drain_slots", 0)
    warmup = config.get("evaluation", {}).get("warmup_slots", 0)
    integer(warmup, "evaluation.warmup_slots", 0)
    if warmup >= sim["slots"]:
        raise ValueError("warmup_slots must be smaller than admission slots")
    number(sim["slot_seconds"], "slot_seconds")
    integer(topo["planes"], "planes")
    integer(topo["sats_per_plane"], "sats_per_plane", 3)
    integer(topo["phase_factor"], "phase_factor", 0)
    integer(topo["max_degree"], "max_degree", 2)
    integer(topo["periodic_change_slots"], "periodic_change_slots")
    if topo["mode"] not in ("walker", "periodic"):
        raise ValueError("topology.mode must be walker or periodic")
    for key in ("altitude_m", "max_isl_distance_m", "link_capacity_bps"):
        number(topo[key], key)
    number(topo["earth_clearance_m"], "earth_clearance_m", strict=False)
    if topo["earth_clearance_m"] >= topo["altitude_m"]:
        raise ValueError("Earth clearance must be below the orbit altitude")
    for key in ("inclination_deg", "crosslink_latitude_limit_deg"):
        number(topo[key], key, strict=False)
        if topo[key] > 90:
            raise ValueError("%s must be <= 90" % key)
    number(tasks["arrival_rate_per_slot"], "arrival_rate_per_slot", strict=False)
    for section, key in ((tasks, "data_bits"), (tasks, "cycles_per_bit"),
                         (tasks, "deadline_seconds"), (config["compute"], "cpu_cycles_per_second")):
        bounds = section[key]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("%s must be [minimum, maximum]" % key)
        for value in bounds:
            number(value, key)
        if bounds[0] > bounds[1]:
            raise ValueError("%s bounds are reversed" % key)
    number(tasks["hotspot_probability"], "hotspot_probability", strict=False)
    if tasks["hotspot_probability"] > 1:
        raise ValueError("hotspot_probability must be <= 1")
    satellites = topo["planes"] * topo["sats_per_plane"]
    if len(set(tasks["hotspot_satellites"])) != len(tasks["hotspot_satellites"]):
        raise ValueError("Hotspot satellite IDs must be unique")
    for satellite in tasks["hotspot_satellites"]:
        integer(satellite, "hotspot_satellite", 0)
        if satellite >= satellites:
            raise ValueError("Hotspot satellite outside constellation")
    if not tasks["hotspot_satellites"] and tasks["hotspot_probability"]:
        raise ValueError("Nonzero hotspot probability needs hotspot satellites")
    for key in ("max_compute_hops", "max_path_hops", "lookahead_slots"):
        integer(routing[key], key, 0)
    for key in ("k_paths", "path_search_limit"):
        integer(routing[key], key)
    integer(routing["path_expansion_limit"], "path_expansion_limit")
    if routing["path_backend"] not in ("bounded", "yen"):
        raise ValueError("routing.path_backend must be bounded or yen")
    if routing["path_search_limit"] < routing["k_paths"]:
        raise ValueError("path_search_limit must be >= k_paths")
    number(routing["reference_data_bits"], "reference_data_bits")
    number(routing["reference_rate_fraction"], "reference_rate_fraction")
    if routing["reference_rate_fraction"] > 1:
        raise ValueError("reference_rate_fraction must be <= 1")
    if config["resource"]["allocation"] not in ("sqrt", "equal"):
        raise ValueError("resource.allocation must be sqrt or equal")
    for section, key in ((sim, "drop_at_deadline"), (routing, "deadline_mask"),
                         (routing, "allow_unverified_future")):
        if not isinstance(section[key], bool):
            raise ValueError("%s must be a boolean" % key)
    for key in ("deadline_penalty", "route_failure_penalty"):
        number(config["reward"][key], key, strict=False)
    number(config["reward"]["normalizer"], "normalizer")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
