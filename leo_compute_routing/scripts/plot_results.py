"""Plot completion delay beside success/failure/censor rates, never alone."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    with args.summary.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No experiment rows in summary")
    names = [row["algorithm"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    delays = [float(row["mean_completion_delay_s"]) if row["mean_completion_delay_s"] else np.nan for row in rows]
    axes[0].bar(names, delays)
    axes[0].set_ylabel("Mean delay of completed tasks (s)")
    x = np.arange(len(names))
    for index, (key, label) in enumerate((("success_rate", "On-time success"),
                                         ("route_failure_rate", "Route failure"), ("censored_rate", "Censored"))):
        axes[1].bar(x + (index - 1) * 0.25, [float(row[key]) for row in rows], width=0.25, label=label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Task fraction (metrics may overlap)")
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    output = args.output or args.summary.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print("Plot:", output.resolve())


if __name__ == "__main__":
    main()
