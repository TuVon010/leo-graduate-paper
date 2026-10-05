"""Plot saved PPO training/validation logs; no smoothing or fabricated points."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from _bootstrap import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with (args.directory / "episodes.csv").open(encoding="utf-8") as stream:
        training = list(csv.DictReader(stream))
    with (args.directory / "validation.csv").open(encoding="utf-8") as stream:
        validation = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in validation:
        grouped[int(row["update"])].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, label in zip(axes, ["total_reward", "success_rate"], ["Episode reward", "On-time success rate"]):
        axis.plot([int(r["update"]) for r in training], [float(r[metric]) for r in training],
                  "o-", markersize=3, alpha=0.6, label="Training episodes")
        updates = sorted(grouped)
        axis.plot(updates, [sum(float(r[metric]) for r in grouped[u]) / len(grouped[u]) for u in updates],
                  "s-", label="Validation seed mean")
        axis.set(xlabel="PPO update", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    path = args.directory / "learning_curve.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(path.resolve())
    diagnostics = [("predicted_invalid_action_rate", "Prediction-invalid selection rate"),
                   ("shield_blocked_probability_mass", "Blocked unmasked probability mass"),
                   ("route_failure_rate", "Actual route failure rate"),
                   ("deadline_violation_rate", "Actual deadline violation rate")]
    if training and all(metric in training[0] for metric, _ in diagnostics):
        figure, axes = plt.subplots(2, 2, figsize=(10, 7))
        for axis, (metric, label) in zip(axes.flat, diagnostics):
            axis.plot([int(r["update"]) for r in training], [float(r[metric]) for r in training],
                      "o-", markersize=3, alpha=0.6, label="Training episodes")
            updates = sorted(u for u in grouped if all(r.get(metric) not in (None, "") for r in grouped[u]))
            if updates:
                axis.plot(updates, [sum(float(r[metric]) for r in grouped[u]) / len(grouped[u]) for u in updates],
                          "s-", label="Validation seed mean")
            axis.set(xlabel="PPO update", ylabel=label)
            axis.grid(alpha=0.2)
            axis.legend()
        figure.tight_layout()
        path = args.directory / "shield_curve.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        print(path.resolve())


if __name__ == "__main__":
    main()
