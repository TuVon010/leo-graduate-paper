"""Progress dashboard and matched-input consolidation for tmux experiments."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def rows(path):
    if not path.exists(): return []
    try:
        with path.open(encoding="utf-8") as stream: return list(csv.DictReader(stream))
    except (OSError, csv.Error): return []


def summarize(root, manifest):
    paths = list((root / "results").glob("*/eval/representations/coupled24/init_2026/seed_metrics.csv"))
    if manifest.get("reference_results"):
        paths += list(Path(manifest["reference_results"]).glob("eval/main/coupled24/init_2026/seed_metrics.csv"))
    records = [r for p in paths for r in rows(p)]
    keys = ("seed", "task_trace_sha256", "cpu_capacities_sha256", "topology_signature")
    if not records or any(len({r[k] for r in records}) != 1 for k in keys):
        raise ValueError("Evaluation inputs differ; refusing to merge an unmatched comparison")
    unique = {}
    for record in records: unique.setdefault(record["algorithm"], record)
    ordered = sorted(unique.values(), key=lambda r: float(r["mean_completion_delay_s"]))
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]));writer.writeheader();writer.writerows(ordered)
    (root / "comparison.json").write_text(json.dumps(dict(evaluation_role="validation_reuse",
        independent_test=False, matching_input_check=True, results=ordered), indent=2), encoding="utf-8")


def snapshot(root):
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    print("Representation experiments | session=%s | commit=%s" % (manifest["session"], manifest["commit"]))
    print("Updated %s | root=%s" % (datetime.now(timezone.utc).isoformat(), root))
    complete = True
    for variant in manifest["variants"]:
        status_path = root / "status" / (variant + ".json")
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        state = status.get("state", "pending");complete &= state == "completed"
        train = root / "results" / variant / "train/coupled24" / variant / "init_2026"
        updates = rows(train / "updates.csv");validation = rows(train / "validation.csv")
        last = updates[-1] if updates else {};val = validation[-1] if validation else {}
        print("%-12s %-10s update=%s/%s FPS=%s val_delay=%s val_success=%s log=%s" % (
            variant, state, last.get("update", 0), manifest["updates"], last.get("rollout_fps", "-"),
            val.get("mean_completion_delay_s", "-"), val.get("success_rate", "-"), status.get("console_log", "-")))
    if complete:
        summarize(root, manifest)
        print("[COMPLETE] matched comparison: " + str(root / "comparison.csv"))
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        if args.watch: print("\033[2J\033[H", end="")
        done = snapshot(args.run_root.resolve())
        if not args.watch: break
        time.sleep(30 if done else 10)


if __name__ == "__main__":
    main()
