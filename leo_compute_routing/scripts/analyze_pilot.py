"""Read-only audit of downloaded server pilots; never selects test-set checkpoints."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return rows


def grouped(rows, key, fields):
    return [{key: item, **{field: float(np.mean([row[field] for row in rows if row[key] == item]))
                          for field in fields}}
            for item in sorted({row[key] for row in rows})]


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {"source": str(root.resolve()), "selection": "validation only; test seed 201 is exploratory",
              "runs": {}, "test": {}, "trace_integrity": {}}
    fields = ["total_reward", "success_rate", "mean_completion_delay_s", "p95_completion_delay_s",
              "mean_cost_per_admitted_task_s", "mean_hops", "route_failure_rate"]
    for folder in sorted((root / "train").glob("*/*/init_*")):
        case, variant = folder.parent.parent.name, folder.parent.name
        episodes, updates, validation = [read_csv(folder / (name + ".csv"))
                                         for name in ("episodes", "updates", "validation")]
        with (folder / "training_summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        # The first and second passes use matching task/CPU seeds; no CI across tasks.
        paired = []
        seeds = sorted({row["seed"] for row in episodes})
        for seed in seeds:
            values = [row for row in episodes if row["seed"] == seed]
            if len(values) >= 2:
                paired.append({"seed": seed, **{field: values[-1][field] - values[0][field] for field in fields}})
        run = {"summary": summary, "validation": grouped(validation, "update", fields),
               "paired_training_pass_deltas": paired,
               "first_five_updates": {key: float(np.mean([row[key] for row in updates[:5]]))
                                      for key in ("value_loss", "gradient_norm", "entropy_per_task", "explained_variance_before_update")},
               "last_five_updates": {key: float(np.mean([row[key] for row in updates[-5:]]))
                                     for key in ("value_loss", "gradient_norm", "entropy_per_task", "explained_variance_before_update")},
               "mean_update_seconds": float(np.mean([row["update_wall_seconds"] for row in updates]))}
        report["runs"][case + "/" + variant] = run

    for folder in sorted((root / "eval/main").glob("*/init_*")):
        rows = read_csv(folder / "seed_metrics.csv")
        case = folder.parent.name
        report["test"][case] = [{key: row[key] for key in ["algorithm", "seed", "task_count", *fields,
                                                         "cpu_utilization", "link_utilization", "censored_rate"]} for row in rows]
        report["trace_integrity"][case] = {key: len({row[key] for row in rows}) == 1
                                           for key in ("task_trace_sha256", "cpu_capacities_sha256", "topology_signature")}

    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    cases = sorted(report["test"])
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 4 * len(cases)), squeeze=False)
    colors = {"full": "#2676AE", "mlp": "#D89032"}
    for i, case in enumerate(cases):
        for variant, color in colors.items():
            folder = next((root / "train" / case / variant).glob("init_*"))
            rows = read_csv(folder / "episodes.csv")
            points = grouped(rows, "update", ["mean_cost_per_admitted_task_s"])
            axes[i, 0].plot([r["update"] for r in points], [r["mean_cost_per_admitted_task_s"] for r in points],
                            alpha=.28, color=color)
            values = np.array([r["mean_cost_per_admitted_task_s"] for r in points])
            if len(points) >= 5:
                axes[i, 0].plot([r["update"] for r in points[4:]], np.convolve(values, np.ones(5)/5, "valid"),
                                color=color, label=variant + " train (5-update mean)")
            val = report["runs"][case + "/" + variant]["validation"]
            axes[i, 0].plot([r["update"] for r in val], [r["mean_cost_per_admitted_task_s"] for r in val],
                            "o--", color=color, label=variant + " validation")
            updates = read_csv(folder / "updates.csv")
            axes[i, 1].plot([r["update"] for r in updates], [r["explained_variance_before_update"] for r in updates],
                            color=color, label=variant)
        test = report["test"][case]
        selected = [r for r in test if r["algorithm"] not in ("local", "shortest_offload")]
        labels = [r["algorithm"].replace("computing_aware", "CA").replace("batch_greedy", "BG").replace("_future", "+F")
                  for r in selected]
        axes[i, 2].bar(labels, [r["mean_cost_per_admitted_task_s"] for r in selected],
                       color=["#2676AE" if r["algorithm"] == "gat_ppo" else "#ABB9C6" for r in selected])
        axes[i, 0].set(title=case + ": cost (lower is better)", xlabel="Update", ylabel="Cost / admitted task (s)")
        axes[i, 0].legend(fontsize=8)
        axes[i, 1].set(title=case + ": critic explained variance", xlabel="Update", ylabel="Explained variance")
        axes[i, 1].axhline(0, color="gray", lw=.8)
        axes[i, 1].legend()
        axes[i, 2].set(title=case + ": exploratory test seed 201", ylabel="Cost / admitted task (s)")
        axes[i, 2].tick_params(axis="x", rotation=30, labelsize=9)
        for axis in axes[i]:
            axis.grid(axis="y", alpha=.2)
    fig.suptitle("20-update pilot: limited validation checkpoints; no convergence claim", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "pilot_diagnosis.png", dpi=180)
    plt.close(fig)
    for name, run in report["runs"].items():
        print(name, "best_update=" + str(run["summary"]["best_update"]),
              "validation=" + json.dumps(run["validation"]),
              "last_critic=" + json.dumps(run["last_five_updates"]))
    print("Audit and plot: " + str(output.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
