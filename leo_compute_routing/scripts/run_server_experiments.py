"""Sequential, resumable server experiment commands; no training logic here.

Use --dry-run to inspect commands without creating directories or importing torch.
All learned variants are independently trained and selected on validation seeds.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter

from _bootstrap import PROJECT_ROOT
from leo_routing.config import fingerprint, load_config


CASES = {
    "coupled24": "configs/experiments/contact24_ppo.yaml",
    "compute24": "configs/experiments/contact_compute_heavy_ppo.yaml",
    "link24": "configs/experiments/contact_link_heavy_ppo.yaml",
    "contact66": "configs/experiments/contact_dynamic_ppo.yaml",
    "scale48": "configs/experiments/contact48_ppo.yaml",
    "smoke": "configs/contact_smoke.yaml",
}
VARIANTS = {
    "full": [],
    "mlp": ["rl.encoder=mlp"],
    "no_shield": ["rl.shield_mode=none"],
    "static_mask": ["rl.shield_mode=mask"],
    "ksp": ["routing.candidate_generation=ksp"],
    "no_future": ["rl.use_future=false"],
    "no_booking": ["rl.use_reservations=false"],
    "equal": ["resource.allocation=equal"],
}
SUITES = {
    "main": ["full", "mlp"],
    "shield": ["full", "no_shield", "static_mask"],
    "modules": ["full", "ksp", "no_future", "no_booking"],
    "resource": ["full", "equal"],
    "scale": ["full", "mlp"],
    "all": list(VARIANTS),
    "audit": [],
    "sensitivity": ["full", "mlp"],
}
BASELINES = ["local", "shortest_offload", "least_load", "computing_aware",
             "computing_aware_future", "batch_greedy", "batch_greedy_future"]


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def execute(command, root, name, dry_run=False, inputs=()):
    """Keep child output outside its data directory; only mark successful calls."""
    print(shlex.join([str(part) for part in command]), flush=True)
    if dry_run:
        return
    signature = {"argv": [str(part) for part in command], "inputs": {}, "configurations": {}}
    for option in ("--config", "--configs"):
        if option not in command:
            continue
        start = command.index(option) + 1
        for part in command[start:]:
            if str(part).startswith("--"):
                break
            signature["configurations"][str(part)] = fingerprint(load_config(part))
            if option == "--config":
                break
    for path in inputs:
        with Path(path).open("rb") as stream:
            signature["inputs"][str(Path(path).resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    marker = root / "runner_status" / (name + ".json")
    if marker.exists():
        if read_json(marker)["signature"] != signature:
            raise ValueError("Completed job arguments changed; choose a new output root: " + name)
        print("Already completed: " + name, flush=True)
        return
    log = root / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise ValueError("An unfinished job has a log. Preserve it and use a new output root: " + str(log))
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    started = perf_counter()
    print("Log: " + str(log), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(part) for part in command], cwd=PROJECT_ROOT, env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    elapsed = perf_counter() - started
    write_json(marker, {"signature": signature, "elapsed_seconds": elapsed})
    print("Completed %s in %.1f s" % (name, elapsed), flush=True)


def training_path(root, case, variant, initialization):
    return root / "train" / case / variant / ("init_%s" % initialization)


def train_case(args, case, variants):
    config_path = PROJECT_ROOT / CASES[case]
    for initialization in args.initializations:
        for variant in variants:
            output = training_path(args.output, case, variant, initialization)
            overrides = ["rl.seed=%s" % initialization, "rl.updates=%s" % args.updates,
                         "rl.device=%s" % args.device] + VARIANTS[variant]
            expected = load_config(config_path, overrides)
            if not args.dry_run and output.exists() and any(output.iterdir()):
                summary = output / "training_summary.json"
                if (summary.exists() and (output / "best.pt").exists() and (output / "last.pt").exists()
                        and read_json(summary)["updates"] == args.updates
                        and fingerprint(load_config(output / "resolved_config.yaml")) == fingerprint(expected)):
                    print("Reuse completed training: " + str(output), flush=True)
                    continue
                raise ValueError("Incomplete or different training: %s. Use a new --output with --resume-root %s"
                                 % (output, args.output))
            command = [sys.executable, "-u", "scripts/train_ppo.py", "--config", config_path, "--output", output]
            for override in overrides:
                command += ["--set", override]
            inputs = []
            if args.resume_root:
                source = training_path(args.resume_root, case, variant, initialization) / "last.pt"
                if source.exists():
                    command += ["--resume", source]
                    inputs.append(source)
                else:
                    print("No source checkpoint; start fresh: " + str(source), flush=True)
            execute(command, args.output, "train/%s/%s/init_%s" % (case, variant, initialization),
                    args.dry_run, inputs)
            execute([sys.executable, "scripts/plot_training.py", output], args.output,
                    "plot/%s/%s/init_%s" % (case, variant, initialization), args.dry_run)


def evaluate_case(args, case, variants, label=None, config=None, algorithms=None, own_config=False):
    label = label or args.suite
    for initialization in args.initializations:
        checkpoints = [training_path(args.output, case, variant, initialization) / "best.pt" for variant in variants]
        if not args.dry_run and any(not path.exists() for path in checkpoints):
            raise ValueError("Train missing checkpoints first: " + str(checkpoints))
        output = args.output / "eval" / label / case / ("init_%s" % initialization)
        command = [sys.executable, "-u", "scripts/evaluate_ppo.py", "--checkpoints", *checkpoints,
                   "--seeds", *args.test_seeds, "--device", args.eval_device, "--output", output,
                   "--bootstrap-samples", "5000", "--algorithms",
                   *(args.algorithms if algorithms is None else algorithms)]
        if not own_config:
            command += ["--config", config or PROJECT_ROOT / CASES[case]]
        execute(command, args.output, "eval/%s/%s/init_%s" % (label, case, initialization),
                args.dry_run, checkpoints)
        if not args.dry_run:
            names = read_json(output / "evaluation_study.json")["learned_policies"]
            write_json(output / "variant_mapping.json", dict(zip(names, variants)))


def generalize(args):
    for initialization in args.initializations:
        checkpoints = [training_path(args.output, "scale48", v, initialization) / "best.pt" for v in ("full", "mlp")]
        command = [sys.executable, "-u", "scripts/run_generalization.py", "--checkpoints", *checkpoints,
                   "--configs", *[PROJECT_ROOT / ("configs/experiments/contact%s_ppo.yaml" % n) for n in (24, 48, 72, 96)],
                   "--seeds", *args.test_seeds, "--device", args.eval_device, "--algorithms", *args.algorithms,
                   "--output", args.output / "generalization" / ("init_%s" % initialization)]
        execute(command, args.output, "generalization/init_%s" % initialization, args.dry_run, checkpoints)


def sensitivity(args, case):
    """Frozen robustness sweeps; do not train or select models on these seeds."""
    import yaml
    base = load_config(PROJECT_ROOT / CASES[case])
    grid = {
        "load": [(str(factor), {"tasks": {"arrival_rate_per_slot": base["tasks"]["arrival_rate_per_slot"] * factor}})
                 for factor in (0.5, 1.0, 1.5)],
        "cpu": [(str(factor), {"compute": {"cpu_cycles_per_second": [f * factor for f in base["compute"]["cpu_cycles_per_second"]]}})
                for factor in (0.5, 1.0, 1.5)],
        "bandwidth": [(str(factor), {"topology": {"link_capacity_bps": base["topology"]["link_capacity_bps"] * factor}})
                      for factor in (0.5, 1.0, 2.0)],
        "deadline": [(str(factor), {"tasks": {"deadline_seconds": [t * factor for t in base["tasks"]["deadline_seconds"]]}})
                     for factor in (0.75, 1.0, 1.25)],
        "lookahead": [(str(slots), {"routing": {"lookahead_slots": slots}}) for slots in (8, 16, 32)],
    }
    for parameter, points in grid.items():
        for value, patch in points:
            resolved = deepcopy(base)
            for section, changes in patch.items():
                resolved[section].update(changes)
            path = args.output / "sensitivity_configs" / case / (parameter + "_" + value + ".yaml")
            if not args.dry_run:
                if path.exists() and fingerprint(load_config(path)) != fingerprint(resolved):
                    raise ValueError("Sweep config changed: " + str(path))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            evaluate_case(args, case, ["full", "mlp"], "sensitivity/" + parameter + "_" + value, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="main")
    parser.add_argument("--phase", choices=["train", "evaluate", "both"], default="both")
    parser.add_argument("--cases", choices=CASES, nargs="+", default=["compute24", "contact66"])
    parser.add_argument("--initializations", type=int, nargs="+", default=[2026])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[201])
    parser.add_argument("--algorithms", choices=BASELINES, nargs="*", default=BASELINES)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.updates < 1 or any(i < 0 for i in args.initializations + args.test_seeds):
        parser.error("Use positive updates and nonnegative seeds")
    for values in (args.cases, args.initializations, args.test_seeds, args.algorithms):
        if len(values) != len(set(values)):
            parser.error("Duplicate cases, seeds or algorithms")
    if not args.test_seeds:
        parser.error("Use at least one test seed")
    for case in args.cases:
        config = load_config(PROJECT_ROOT / CASES[case])
        if set(args.test_seeds) & set(config["rl"]["train_seeds"] + config["rl"]["validation_seeds"]):
            parser.error("Test seeds overlap training/validation")
    args.output = args.output.resolve()
    if args.resume_root:
        args.resume_root = args.resume_root.resolve()
        if args.resume_root == args.output:
            parser.error("Resume into a NEW output root; preserve the source run")
    if args.suite == "scale":
        if args.cases != ["scale48"]:
            parser.error("Use --suite scale --cases scale48")
    if args.suite == "sensitivity" and args.phase != "evaluate":
        parser.error("Frozen sweeps require --phase evaluate and existing main checkpoints")
    if args.suite == "audit":
        for case in args.cases:
            execute([sys.executable, "-u", "scripts/audit_route_exposure.py", "--config", PROJECT_ROOT / CASES[case],
                     "--seeds", "50", "--max-tasks", "256", "--output", args.output / "audit" / case],
                    args.output, "audit/" + case, args.dry_run)
        return
    if args.suite == "sensitivity":
        for case in args.cases:
            sensitivity(args, case)
        return
    variants = SUITES[args.suite]
    for case in args.cases:
        if args.phase != "evaluate":
            train_case(args, case, variants)
        if args.phase != "train" and args.suite != "scale":
            common = [v for v in variants if v != "equal"]
            evaluate_case(args, case, common)
            if "equal" in variants:
                # Never silently apply sqrt to a checkpoint trained with equal sharing.
                # Both resource variants replay the same seeds using their own allocator.
                for variant in ("full", "equal"):
                    evaluate_case(args, case, [variant], "resource_own/" + variant,
                                  algorithms=[], own_config=True)
    if args.suite == "scale" and args.phase != "train":
        generalize(args)


if __name__ == "__main__":
    main()
