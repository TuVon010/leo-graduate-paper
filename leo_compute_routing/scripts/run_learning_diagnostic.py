"""Short orbital validation-only numerical diagnostic, NOT a convergence study.

Compare raw critic, scaled critic and scaled+prior under identical small budgets.
Full-length server training and independent test are still required.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.utils.console import capture_console, console_log_path
from leo_routing.utils.io import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--slots", type=int, default=128)
    args = parser.parse_args()
    if args.updates < 1 or args.slots < 1:
        parser.error("updates and slots must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new")
    with capture_console(console_log_path(args.output, "learning_diagnostic")) as log:
        from leo_routing.agents.trainer import train_ppo
        from leo_routing.agents.ppo_agent import PPOAgent
        from leo_routing.evaluation.evaluator import run_comparison
        config = load_config(PROJECT_ROOT / "configs/experiments/contact_compute_tuned.yaml")
        config["simulation"].update(slots=args.slots, drain_slots=160)
        config["rl"].update(updates=args.updates, episodes_per_update=1, epochs=2, device="cpu",
                            validation_seeds=[100], validation_every=1, checkpoint_every=1)
        policies, runs = [], {}
        for name, scale, strength in (("raw", 1.0, 0.0), ("scaled", 100.0, 0.0), ("scaled_prior", 100.0, 2.0)):
            current = deepcopy(config)
            current["rl"].update(value_scale=scale, completion_prior_strength=strength)
            folder = args.output / name
            train_ppo(current, folder, progress=lambda message: print(message, flush=True),
                      log_interval_seconds=10, console_log=log)
            policy, _ = PPOAgent.load(folder / "best.pt", "cpu")
            policy.name = name
            policies.append(policy)
            with (folder / "updates.csv").open(encoding="utf-8") as stream:
                runs[name] = list(csv.DictReader(stream))
        validation = deepcopy(config)
        validation["simulation"]["seed"] = 100
        rows = run_comparison(validation, ["batch_greedy", "batch_greedy_future", "contact_greedy"],
                              args.output / "validation_comparison", extra_policies=policies,
                              progress=lambda message: print(message, flush=True))
        save_json(args.output / "diagnostic.json", {"study": "short orbital validation-only numerical check",
                  "validation_seed": 100, "init_seed": 2026, "training_seed_schedule": config["rl"]["train_seeds"],
                  "slots": args.slots, "updates": args.updates, "test_set_evaluated": False,
                  "convergence_demonstrated": False, "runs": runs, "validation_metrics": rows})
        for row in rows:
            print("[DIAGNOSTIC]", row["algorithm"], "delay=", row["mean_completion_delay_s"],
                  "success=", row["success_rate"], "cost=", row["mean_cost_per_admitted_task_s"], flush=True)
        print("Report:", (args.output / "diagnostic.json").resolve())


if __name__ == "__main__":
    main()
