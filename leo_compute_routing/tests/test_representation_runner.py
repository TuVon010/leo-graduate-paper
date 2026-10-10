import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


def test_tmux_dry_run_does_not_create_output(tmp_path, monkeypatch, capsys):
    launcher = module("launch_representation_tmux")
    root = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["launcher", "--run-root", str(root), "--session", "test-01", "--dry-run"])
    launcher.main()
    windows = json.loads(capsys.readouterr().out)
    assert set(windows) == {"self_graph", "mlp_kkt", "gat_kkt", "gated_kkt"}
    assert not root.exists()


def write_metrics(root, variant, trace="shared"):
    path = root / "results" / variant / "eval/representations/coupled24/init_2026/seed_metrics.csv"
    path.parent.mkdir(parents=True)
    record = dict(algorithm=variant, seed="100", task_trace_sha256=trace,
                  cpu_capacities_sha256="cpu", topology_signature="topo", mean_completion_delay_s="1.0")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record));writer.writeheader();writer.writerow(record)


def test_consolidation_requires_matching_external_inputs(tmp_path):
    status = module("representation_status")
    write_metrics(tmp_path, "mlp_kkt");write_metrics(tmp_path, "gated_kkt")
    status.summarize(tmp_path, {})
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert result["matching_input_check"] and not result["independent_test"]
    assert len(result["results"]) == 2
    path = tmp_path / "results/gated_kkt/eval/representations/coupled24/init_2026/seed_metrics.csv"
    path.write_text(path.read_text().replace("shared", "different"))
    with pytest.raises(ValueError, match="inputs differ"):
        status.summarize(tmp_path, {})


def test_window_failure_is_logged_and_not_marked_completed(tmp_path, monkeypatch):
    window = module("run_representation_window")
    monkeypatch.setattr(window, "job_command", lambda *a: [sys.executable, "-u", "-c",
        "import os,sys; print(os.environ['TMPDIR'],flush=True); print('FAILED_CHILD',flush=True); sys.exit(7)"])
    monkeypatch.setattr(sys, "argv", ["window", "--run-root", str(tmp_path), "--variant", "gated_kkt", "--updates", "1"])
    with pytest.raises(SystemExit) as error:
        window.main()
    assert error.value.code == 7
    record = json.loads((tmp_path / "status/gated_kkt.json").read_text())
    assert record["state"] == "failed" and record["returncode"] == 7
    log = (tmp_path / "console/gated_kkt.log").read_text()
    assert "FAILED_CHILD" in log and str(tmp_path / "tmp") in log
