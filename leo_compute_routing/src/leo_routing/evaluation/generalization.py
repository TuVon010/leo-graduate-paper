"""Frozen-checkpoint cross-topology evaluation with an in-domain held-out anchor."""
from copy import deepcopy
from pathlib import Path

from ..agents.ppo_agent import PPOAgent
from ..config import fingerprint, validate_config
from ..utils.io import save_csv, save_json
from .rl_evaluator import evaluate_checkpoints
from .statistics import bootstrap_interval


def physical_fingerprint(config):
    physical = deepcopy(config)
    physical.pop("rl", None)
    physical["simulation"].pop("seed", None)
    return fingerprint(physical)


def run_generalization(checkpoints, targets, seeds, output_directory, algorithms=(), device="cpu",
                       bootstrap_samples=5000, progress=None, log_interval_seconds=10.0):
    checkpoints, seeds, targets = list(checkpoints), list(seeds), dict(targets)
    if not checkpoints or not targets:
        raise ValueError("Provide checkpoints and named target configurations")
    policies = [PPOAgent.load(path, device)[0] for path in checkpoints]
    anchor = policies[0].config
    if any(physical_fingerprint(policy.config) != physical_fingerprint(anchor) for policy in policies):
        raise ValueError("Compared checkpoints must share the same training physical configuration")
    for name, config in targets.items():
        if not name or Path(name).name != name or name in (".", "..", "in_domain"):
            raise ValueError("Invalid target name")
        validate_config(config)
    if anchor.get("evaluation", {}).get("warmup_slots", 0) or any(
            c.get("evaluation", {}).get("warmup_slots", 0) for c in targets.values()):
        raise ValueError("Per-admitted-task cost uses the full episode; use warmup_slots=0 for this study")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "study.json", {
        "checkpoints": [str(Path(p).resolve()) for p in checkpoints], "seeds": seeds,
        "training_satellites": anchor["topology"]["planes"] * anchor["topology"]["sats_per_plane"],
        "targets": {name: {"config_sha256": fingerprint(config),
                    "satellites": config["topology"]["planes"] * config["topology"]["sats_per_plane"]}
                    for name, config in targets.items()},
        "weights_frozen": True, "feature_scales_from_training_checkpoint": True,
        "in_domain_anchor": "same checkpoint on held-out training-configuration seeds, not training reward",
        "cost": "-episode reward * fixed reward normalizer / admitted task count",
        "gap": "(target cost - in-domain held-out cost) / abs(in-domain held-out cost)",
        "gap_limitation": "Includes target intrinsic difficulty; not a pure generalization error. "
                          "No target-specific retraining or checkpoint selection is performed."})
    if progress:
        progress("Evaluating in-domain held-out anchor ...")
    anchor_rows = evaluate_checkpoints(checkpoints, seeds, output / "in_domain", algorithms, anchor,
                                      device, bootstrap_samples, progress, log_interval_seconds)
    anchor_index = {(row["algorithm"], row["seed"]): row for row in anchor_rows}
    rows, gaps = [], []
    for name, config in targets.items():
        same = physical_fingerprint(config) == physical_fingerprint(anchor)
        if progress:
            progress("Target %s%s" % (name, " (reusing identical in-domain evaluation)" if same else ""))
        target_rows = anchor_rows if same else evaluate_checkpoints(checkpoints, seeds, output / name,
            algorithms, config, device, bootstrap_samples, progress, log_interval_seconds)
        for row in target_rows:
            cost = row["mean_cost_per_admitted_task_s"]
            reference = anchor_index[(row["algorithm"], row["seed"])]["mean_cost_per_admitted_task_s"]
            gap = (cost - reference) / abs(reference) if cost is not None and reference not in (None, 0) else None
            rows.append({"target": name, "satellites": config["topology"]["planes"] * config["topology"]["sats_per_plane"],
                         **row, "in_domain_cost_s": reference, "relative_cost_shift": gap})
        save_csv(output / "seed_metrics.csv", rows)
        for algorithm in sorted({row["algorithm"] for row in target_rows}):
            values = [row["relative_cost_shift"] for row in rows if row["target"] == name and row["algorithm"] == algorithm]
            mean, low, high = bootstrap_interval(values, bootstrap_samples)
            gaps.append({"target": name, "algorithm": algorithm, "seed_count": len(values),
                         "relative_cost_shift": mean, "ci95_low": low, "ci95_high": high})
        save_csv(output / "cost_shift.csv", gaps)
    return rows
