"""Checkpoint comparisons, with explicitly labeled validation reuse for debugging."""
from copy import deepcopy
import hashlib
from pathlib import Path

from ..agents.ppo_agent import PPOAgent
from ..utils.io import save_csv, save_json
from .evaluator import run_comparison
from .statistics import aggregate_seed_results


def evaluate_checkpoints(checkpoints, seeds, output_directory, algorithms=(), config=None,
                         device="cpu", bootstrap_samples=5000, progress=None, log_interval_seconds=10.0,
                         allow_validation_reuse=False):
    seeds, checkpoints, algorithms = list(seeds), list(checkpoints), list(algorithms)
    if not checkpoints or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply checkpoints and at least one distinct evaluation seed")
    if len(set(algorithms)) != len(algorithms) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Invalid seeds or duplicate baseline names")
    policies = []
    used_names = set(algorithms)
    reused_by_policy = {}
    for path in checkpoints:
        policy, _ = PPOAgent.load(path, device)
        training_overlap = set(policy.settings["train_seeds"]) & set(seeds)
        validation_overlap = set(policy.settings["validation_seeds"]) & set(seeds)
        if training_overlap:
            raise ValueError("Evaluation seeds overlap checkpoint training seeds: %s" % sorted(training_overlap))
        if validation_overlap and not allow_validation_reuse:
            raise ValueError("Evaluation seeds overlap validation seeds: %s; use --allow-validation-reuse for "
                             "a labeled development evaluation" % sorted(validation_overlap))
        original_name, suffix = policy.name, 2
        while policy.name in used_names:
            policy.name = original_name + "_%s" % suffix
            suffix += 1
        used_names.add(policy.name)
        reused_by_policy[policy.name] = sorted(validation_overlap)
        checksum = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        policy.metadata.update({"checkpoint_path": str(Path(path).resolve()), "checkpoint_sha256": checksum.hexdigest(),
                                "update_count": policy.update_count, "deterministic_evaluation": True,
                                "validation_reuse_seeds": sorted(validation_overlap),
                                "independent_test": not bool(validation_overlap)})
        policies.append(policy)
    configured = deepcopy(config if config is not None else policies[0].config)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output is nonempty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    reused = any(reused_by_policy.values())
    if progress:
        progress("[EVALUATION ROLE] %s | seeds=%s | independent_test=%s" % (
            "validation_reuse (development evaluation)" if reused else "held_out_test",
            seeds, not reused))
    save_json(output / "evaluation_study.json", {"seeds": seeds, "learned_policies": [p.name for p in policies],
              "baselines": algorithms, "held_out_seed_check": not reused,
              "checkpoint_selection_uses_test": reused,
              "evaluation_role": "validation_reuse" if reused else "held_out_test",
              "independent_test": not reused, "training_seed_overlap_check": True,
              "allow_validation_reuse": allow_validation_reuse,
              "validation_reuse_seeds_by_policy": reused_by_policy,
              "checkpoint_selection_uses_evaluation_seeds": reused,
              "evaluation_action": "greedy argmax of each conditional masked distribution",
              "bootstrap_samples": bootstrap_samples,
              "single_seed_pilot": len(seeds) == 1,
              "confidence_intervals_available": len(seeds) >= 2})
    rows = []
    for seed in seeds:
        current = deepcopy(configured)
        current["simulation"]["seed"] = seed
        rows.extend(run_comparison(current, algorithms, output / ("seed_%s" % seed),
                                   progress=progress, extra_policies=policies, log_interval_seconds=log_interval_seconds))
        save_csv(output / "seed_metrics.csv", rows)
    reference = algorithms[0] if algorithms else policies[0].name
    aggregate, paired = aggregate_seed_results(rows, reference, bootstrap_samples)
    save_csv(output / "aggregate.csv", aggregate)
    save_csv(output / "paired_differences.csv", paired)
    save_json(output / "aggregate.json", {"summary": aggregate, "paired_differences": paired})
    return rows
