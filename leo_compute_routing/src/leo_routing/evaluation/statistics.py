"""Equal-weight seed means and paired percentile bootstrap intervals.

Resample whole seeds, never correlated individual tasks. Undefined metrics are
reported explicitly; no CI is produced if any requested seed is undefined.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..config import validate_config
from ..utils.io import save_csv, save_json
from .evaluator import run_comparison

METRICS = ("success_rate", "mean_completion_delay_s", "p95_completion_delay_s",
           "completion_rate", "deadline_violation_rate", "route_failure_rate", "censored_rate",
           "cpu_utilization", "link_utilization", "mean_queue_cycles")


def bootstrap_interval(values, bootstrap_samples=5000, bootstrap_seed=7301):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not values or any(value is None or not np.isfinite(value) for value in values):
        return None, None, None
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    rng = np.random.default_rng(bootstrap_seed)
    # Bound peak memory for large studies.
    means = []
    for start in range(0, bootstrap_samples, 1000):
        indices = rng.integers(len(values), size=(min(1000, bootstrap_samples - start), len(values)))
        means.extend(values[indices].mean(axis=1))
    low, high = np.percentile(means, [2.5, 97.5])
    return mean, float(low), float(high)


def aggregate_seed_results(rows, reference, bootstrap_samples=5000, bootstrap_seed=7301):
    grouped = {}
    for row in rows:
        by_seed = grouped.setdefault(row["algorithm"], {})
        if row["seed"] in by_seed:
            raise ValueError("Duplicate algorithm/seed result")
        by_seed[row["seed"]] = row
    if reference not in grouped:
        raise ValueError("Reference algorithm is absent")
    expected_seeds = set(grouped[reference])
    if any(set(group) != expected_seeds for group in grouped.values()):
        raise ValueError("All algorithms must have the same seed set")
    seeds = sorted(expected_seeds)
    summaries, paired = [], []
    for algorithm, group in grouped.items():
        for metric in METRICS:
            values = [group[seed][metric] for seed in seeds]
            mean, low, high = bootstrap_interval(values, bootstrap_samples, bootstrap_seed)
            valid = int(sum(value is not None and np.isfinite(value) for value in values))
            summaries.append({"algorithm": algorithm, "metric": metric, "seed_count": len(seeds),
                              "valid_seed_count": valid, "mean": mean, "ci95_low": low, "ci95_high": high})
            if algorithm == reference:
                continue
            differences = [None if group[s][metric] is None or grouped[reference][s][metric] is None
                           else group[s][metric] - grouped[reference][s][metric] for s in seeds]
            mean, low, high = bootstrap_interval(differences, bootstrap_samples, bootstrap_seed)
            paired.append({"algorithm": algorithm, "reference": reference, "metric": metric,
                           "seed_count": len(seeds),
                           "valid_seed_count": int(sum(v is not None and np.isfinite(v) for v in differences)),
                           "mean_difference": mean, "ci95_low": low, "ci95_high": high})
    return summaries, paired


def run_multiseed(config, algorithms, seeds, output_directory, reference="batch_greedy",
                  bootstrap_samples=5000, bootstrap_seed=7301, progress=None):
    algorithms, seeds = list(algorithms), list(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two distinct seeds; five or more recommended for a pilot")
    if len(set(algorithms)) != len(algorithms) or reference not in algorithms:
        raise ValueError("Algorithms must be unique and include the paired reference")
    if bootstrap_samples < 1 or bootstrap_seed < 0:
        raise ValueError("Invalid bootstrap settings")
    configs = []
    for seed in seeds:
        resolved = deepcopy(config)
        resolved["simulation"]["seed"] = seed
        validate_config(resolved)
        configs.append(resolved)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty; choose a new directory to preserve raw results")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "study.json", {"schema_version": 1, "seeds": seeds, "algorithms": algorithms,
              "reference": reference, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
              "ci": "95% percentile bootstrap of equally weighted seed means",
              "difference": "algorithm minus reference; bootstrap whole paired seeds",
              "seed_scope": "each seed varies both task arrivals and sampled CPU capacities",
              "warning": "Few-seed CIs can be unstable; pilot statistics do not establish paper conclusions."})
    rows = []
    for resolved in configs:
        seed = resolved["simulation"]["seed"]
        if progress:
            progress("Seed %s" % seed)
        rows.extend(run_comparison(resolved, algorithms, output / ("seed_%s" % seed), progress=progress))
        save_csv(output / "seed_metrics.csv", rows)
    summary, paired = aggregate_seed_results(rows, reference, bootstrap_samples, bootstrap_seed)
    save_csv(output / "aggregate.csv", summary)
    save_csv(output / "paired_differences.csv", paired)
    save_json(output / "aggregate.json", {"summary": summary, "paired_differences": paired})
    return summary, paired
