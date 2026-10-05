import importlib.util
import io
from pathlib import Path
import subprocess
import sys

import pytest

from leo_routing.utils.console import capture_console
from leo_routing.utils.progress import EpisodeProgress


def test_transcript_flushes_both_streams_and_keeps_exception(tmp_path):
    before_out, before_err = sys.stdout, sys.stderr
    path = tmp_path / "console.log"
    with pytest.raises(RuntimeError, match="failure evidence"):
        with capture_console(path):
            print("实时训练 stdout")
            print("stderr evidence", file=sys.stderr)
            saved = path.read_text(encoding="utf-8")
            assert "实时训练 stdout" in saved and "stderr evidence" in saved
            raise RuntimeError("failure evidence")
    assert sys.stdout is before_out and sys.stderr is before_err
    assert "RuntimeError: failure evidence" in path.read_text(encoding="utf-8")


def test_fps_counts_environment_steps_not_tasks(monkeypatch):
    import leo_routing.utils.progress as progress
    times = iter([0.0, 2.0, 4.0])
    monkeypatch.setattr(progress, "perf_counter", lambda: next(times))
    lines = []
    reporter = EpisodeProgress({"simulation": {"slots": 10, "drain_slots": 5, "slot_seconds": 0.25}}, lines.append)
    reporter.step({"slot_metrics": {"arrivals": 100, "active_tasks": 100}}, -3.0)
    metrics = reporter.metrics()
    assert metrics["environment_steps"] == 1
    assert metrics["environment_fps"] == 0.25  # one step / four wall seconds
    assert metrics["admitted_tasks_per_second"] == 25
    assert "FPS=0.50" in lines[0] and "phase=admission" in lines[0]


def load_runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("server_runner_under_test", scripts / "run_server_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_child_output_is_forwarded_before_exit_and_failure_preserved(tmp_path, monkeypatch):
    runner = load_runner(monkeypatch)
    release = tmp_path / "release"

    class LiveOutput(io.StringIO):
        def write(self, text):
            if text.strip() == "CHILD_READY":
                release.touch()
            return super().write(text)

    terminal = LiveOutput()
    monkeypatch.setattr(sys, "stdout", terminal)
    child = ("import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); print('CHILD_READY',flush=True); "
             "deadline=time.monotonic()+3\n"
             "while not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
             "print('释放成功' if p.exists() else 'NO_LIVE_OUTPUT',flush=True)\n"
             "print('CHILD_ERROR',file=sys.stderr,flush=True); sys.exit(3)")
    with pytest.raises(subprocess.CalledProcessError) as error:
        with capture_console(tmp_path / "all_console.log"):
            runner.execute([sys.executable, "-u", "-c", child, release], tmp_path / "run", "failure_test")
    assert error.value.returncode == 3 and release.exists()
    output = terminal.getvalue()
    assert "释放成功" in output and "CHILD_ERROR" in output
    assert "NO_LIVE_OUTPUT\n" not in (tmp_path / "run/logs/failure_test.log").read_text(encoding="utf-8")
    assert "CHILD_ERROR" in (tmp_path / "all_console.log").read_text(encoding="utf-8")
    assert not (tmp_path / "run/runner_status/failure_test.json").exists()
