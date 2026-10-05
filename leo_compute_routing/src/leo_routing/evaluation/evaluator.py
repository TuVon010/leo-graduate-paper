from copy import deepcopy
from dataclasses import asdict
import platform
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import yaml

from .. import __version__
from ..baselines import make_policy
from ..config import fingerprint
from ..env.leo_env import LeoEnv, generate_cpu_capacities
from ..tasks.task_generator import generate_task_trace, save_task_trace
from ..topology.topology_cache import get_topology
from ..utils.io import save_csv, save_json, save_yaml
from .calibration import audit_scenario
from ..agents.diagnostics import add_counts, decision_metrics
from ..utils.progress import EpisodeProgress, device_summary, outcome_text


def evaluate_policy(config, topology, task_trace, cpu_capacities, policy, progress=None,
                    log_interval_seconds=10.0, phase="EVAL"):
    policy_config = deepcopy(config)
    # Snapshot baselines receive no future features or future capacity estimates.
    if hasattr(policy, "configure_environment"):
        policy_config = policy.configure_environment(policy_config)
    elif not policy.use_future:
        policy_config["routing"]["lookahead_slots"] = 0
        policy_config["routing"]["deadline_mask"] = False
    environment = LeoEnv(policy_config, topology, task_trace, cpu_capacities)
    reporter = EpisodeProgress(config, progress, "%s %s seed=%s" % (phase, policy.name, config["simulation"]["seed"]),
                               log_interval_seconds)
    observation, _ = environment.reset()
    decision_seconds, decision_tasks = 0.0, 0
    diagnostics = {}
    while True:
        decision_start = perf_counter()
        actions = policy.select(observation)
        if hasattr(policy, "last_decision_metrics"):
            add_counts(diagnostics, policy.last_decision_metrics)
        duration = perf_counter() - decision_start
        if observation.tasks:
            decision_seconds += duration
            decision_tasks += len(observation.tasks)
        observation, reward, terminated, truncated, info = environment.step(actions)
        reporter.step(info, reward)
        if terminated or truncated:
            break
    metrics = {"algorithm": policy.name, "seed": config["simulation"]["seed"], **info["episode_metrics"],
               "policy_select_ms_per_task": 1000 * decision_seconds / decision_tasks if decision_tasks else 0.0,
               "observation_build_seconds": environment.observation_build_seconds,
               "terminated": terminated, "truncated": truncated, **reporter.metrics()}
    admitted = metrics["all_admitted_task_count"]
    metrics["mean_cost_per_admitted_task_s"] = (-metrics["total_reward"] * config["reward"]["normalizer"] / admitted
                                               if admitted else None)
    if diagnostics:
        metrics.update(decision_metrics(diagnostics))
    return metrics, environment


def run_comparison(config, algorithms, output_directory, topology_cache=None, task_trace=None, progress=None,
                   extra_policies=None, log_interval_seconds=10.0):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    # Validate policy names before generating expensive traces.
    policies = [make_policy(name) for name in algorithms] + list(extra_policies or ())
    if not policies or len({policy.name for policy in policies}) != len(policies):
        raise ValueError("Comparison requires nonempty, uniquely named policies")
    if progress:
        progress("[EVAL SETUP] seed=%s | result_dir=%s | preparing topology/traces/calibration ..." % (
            config["simulation"]["seed"], output_directory.resolve()))
    topology = get_topology(config, topology_cache, progress, log_interval_seconds)
    tasks = task_trace if task_trace is not None else generate_task_trace(config)
    capacities = generate_cpu_capacities(config)
    calibration = audit_scenario(config, topology, tasks, capacities)
    save_json(output_directory / "calibration.json", calibration)
    if progress:
        for warning in calibration["warnings"]:
            progress("Calibration: " + warning)
    task_hash = fingerprint([[asdict(task) for task in batch] for batch in tasks])
    cpu_hash = fingerprint(capacities.tolist())
    save_yaml(output_directory / "resolved_config.yaml", config)
    topology.save(output_directory / "topology.npz")
    save_task_trace(tasks, output_directory / "task_trace.json")
    save_json(output_directory / "cpu_capacities.json", capacities.tolist())
    diagnostics = topology.diagnostics(config["simulation"]["slots"])
    save_json(output_directory / "manifest.json", {
        "schema_version": 2, "package_version": __version__, "config_sha256": fingerprint(config),
        "task_trace_sha256": task_hash, "cpu_capacities_sha256": cpu_hash,
        "topology_signature": topology.signature, "topology_diagnostics": diagnostics,
        "algorithms": [policy.name for policy in policies], "python": platform.python_version(),
        "learned_policies": {p.name: getattr(p, "metadata", {}) for p in extra_policies or ()},
        "dependencies": {"numpy": np.__version__, "networkx": nx.__version__, "PyYAML": yaml.__version__},
        "model": "piecewise-snapshot fluid store-and-forward with processor sharing",
        "candidate_generation": config["routing"].get("candidate_generation", "ksp"),
        "contact_waiting": "no waiting across unavailable contacts",
        "link_duplex": "shared bidirectional budget", "prediction_is_guarantee": False,
        "utilization_window": "fixed post-warmup admission interval; drain excluded",
        "task_cohort": "arrivals in measurement interval; outcomes followed through drain",
        "warmup_slots": config.get("evaluation", {}).get("warmup_slots", 0),
        "calibration_file": "calibration.json"})
    summaries = []
    for index, policy in enumerate(policies, 1):
        if progress:
            progress("[EVAL] policy=%s (%s/%s) seed=%s starting ..." % (
                policy.name, index, len(policies), config["simulation"]["seed"]))
            if hasattr(policy, "device"):
                progress("[DEVICE] " + device_summary(policy.device))
        metrics, environment = evaluate_policy(config, topology, tasks, capacities, policy, progress, log_interval_seconds)
        metrics.update({"task_trace_sha256": task_hash, "cpu_capacities_sha256": cpu_hash,
                        "topology_signature": topology.signature})
        summaries.append(metrics)
        save_csv(output_directory / policy.name / "tasks.csv", environment.engine.task_records())
        save_csv(output_directory / policy.name / "slots.csv", environment.slot_metrics)
        save_json(output_directory / policy.name / "metrics.json", metrics)
        if progress:
            progress("[EVAL DONE] %s | %s | FPS=%.2f env_steps=%s | data=%s" % (
                policy.name, outcome_text(metrics), metrics["environment_fps"], metrics["environment_steps"],
                (output_directory / policy.name).resolve()))
    save_csv(output_directory / "summary.csv", summaries)
    save_json(output_directory / "summary.json", summaries)
    return summaries
