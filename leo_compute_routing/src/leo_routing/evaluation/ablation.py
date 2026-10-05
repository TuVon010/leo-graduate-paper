from copy import deepcopy
from pathlib import Path

from ..utils.io import save_csv
from .evaluator import run_comparison


def run_ablation(config, output_directory, progress=None):
    """Simulator ablations only; no claim to test an unimplemented GAT/PPO."""
    variants = ("full_heuristic", "without_future_topology", "equal_resource", "without_deadline_mask")
    all_rows = []
    for name in variants:
        variant = deepcopy(config)
        algorithm = "computing_aware_future"
        if name == "without_future_topology":
            # Keep deadline filtering and effective-mask fallback identical;
            # change only the future-contact window to isolate look-ahead.
            variant["routing"]["lookahead_slots"] = 0
        elif name == "equal_resource":
            variant["resource"]["allocation"] = "equal"
        elif name == "without_deadline_mask":
            variant["routing"]["deadline_mask"] = False
        rows = run_comparison(variant, [algorithm], Path(output_directory) / name, progress=progress)
        all_rows.extend({"variant": name, **row} for row in rows)
    save_csv(Path(output_directory) / "ablation.csv", all_rows)
    return all_rows
