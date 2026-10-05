"""Held-out checkpoint comparisons through the SAME simulator and trace writer."""
from copy import deepcopy
import hashlib
from pathlib import Path

from ..agents.ppo_agent import PPOAgent
from ..utils.io import save_csv, save_json
from .evaluator import run_comparison
from .statistics import aggregate_seed_results


def evaluate_checkpoints(checkpoints, seeds, output_directory, algorithms=(), config=None,
                         device="cpu", bootstrap_samples=5000, progress=None):
    seeds, checkpoints, algorithms = list(seeds), list(checkpoints), list(algorithms)
    if not checkpoints or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply checkpoints and at least one distinct held-out seed")
    if len(set(algorithms)) != len(algorithms) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Invalid seeds or duplicate baseline names")
    policies = []
    used_names = set(algorithms)
    for path in checkpoints:
        policy, _ = PPOAgent.load(path, device)
        seen = set(policy.settings["train_seeds"]) | set(policy.settings["validation_seeds"])
        if seen & set(seeds):
            raise ValueError("Test seeds overlap checkpoint training/validation seeds: %s" % sorted(seen & set(seeds)))
        original_name, suffix = policy.name, 2
        while policy.name in used_names:
            policy.name = original_name + "_%s" % suffix
            suffix += 1
        used_names.add(policy.name)
        checksum = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        policy.metadata.update({"checkpoint_path": str(Path(path).resolve()), "checkpoint_sha256": checksum.hexdigest(),
                                "update_count": policy.update_count, "deterministic_evaluation": True})
        policies.append(policy)
    configured = deepcopy(config if config is not None else policies[0].config)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output is nonempty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "evaluation_study.json", {"seeds": seeds, "learned_policies": [p.name for p in policies],
              "baselines": algorithms, "held_out_seed_check": True, "checkpoint_selection_uses_test": False,
              "evaluation_action": "greedy argmax of each conditional masked distribution",
              "bootstrap_samples": bootstrap_samples,
              "single_seed_pilot": len(seeds) == 1,
              "confidence_intervals_available": len(seeds) >= 2})
    rows = []
    for seed in seeds:
        current = deepcopy(configured)
        current["simulation"]["seed"] = seed
        rows.extend(run_comparison(current, algorithms, output / ("seed_%s" % seed),
                                   progress=progress, extra_policies=policies))
        save_csv(output / "seed_metrics.csv", rows)
    reference = algorithms[0] if algorithms else policies[0].name
    aggregate, paired = aggregate_seed_results(rows, reference, bootstrap_samples)
    save_csv(output / "aggregate.csv", aggregate)
    save_csv(output / "paired_differences.csv", paired)
    save_json(output / "aggregate.json", {"summary": aggregate, "paired_differences": paired})
    return rows
