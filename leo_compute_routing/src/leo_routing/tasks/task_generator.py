from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .task import Task


def generate_task_trace(config):
    """Use an isolated stream; policy decisions cannot consume arrival randomness."""
    rng = np.random.default_rng(np.random.SeedSequence([config["simulation"]["seed"], 1001]))
    tasks, trace, task_id = config["tasks"], [], 0
    satellites = config["topology"]["planes"] * config["topology"]["sats_per_plane"]
    for slot in range(config["simulation"]["slots"]):
        batch = []
        for _ in range(rng.poisson(tasks["arrival_rate_per_slot"])):
            if rng.random() < tasks["hotspot_probability"]:
                source = int(rng.choice(tasks["hotspot_satellites"]))
            else:
                source = int(rng.integers(satellites))
            batch.append(Task(task_id, source, float(rng.uniform(*tasks["data_bits"])),
                              float(rng.uniform(*tasks["cycles_per_bit"])),
                              float(rng.uniform(*tasks["deadline_seconds"])), slot))
            task_id += 1
        trace.append(tuple(batch))
    return tuple(trace)


def save_task_trace(trace, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump([[asdict(task) for task in batch] for batch in trace], stream, indent=2)


def load_task_trace(path):
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    trace = tuple(tuple(Task(**task) for task in batch) for batch in raw)
    ids = [task.task_id for batch in trace for task in batch]
    if len(ids) != len(set(ids)):
        raise ValueError("Task trace contains duplicate IDs")
    if any(task.arrival_slot != slot for slot, batch in enumerate(trace) for task in batch):
        raise ValueError("Task trace contains inconsistent arrival slots")
    return trace
