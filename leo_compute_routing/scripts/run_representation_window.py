"""Run one isolated training/evaluation job and preserve its terminal output."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def save_status(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def job_command(python, variant, updates, output):
    return [str(python), "-u", str(ROOT / "scripts/run_server_experiments.py"),
            "--suite", "representations", "--variants", variant, "--phase", "both",
            "--cases", "coupled24", "--updates", str(updates), "--initializations", "2026",
            "--device", "cuda", "--eval-device", "cpu", "--output", str(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if args.variant not in ("full", "mlp", "self_graph", "mlp_kkt", "gat_kkt", "gated_kkt"):
        parser.error("Unknown representation variant")
    output = run_root / "results" / args.variant
    log = run_root / "console" / (args.variant + ".log")
    status = run_root / "status" / (args.variant + ".json")
    for path in (log.parent, status.parent):
        path.mkdir(parents=True, exist_ok=True)
    if log.exists() or status.exists():
        parser.error("Window output exists; use a new run root")
    environment = os.environ.copy()
    for name, relative in {"TMPDIR": "tmp", "PYTHONPYCACHEPREFIX": "cache/pycache",
                           "MPLCONFIGDIR": "cache/matplotlib", "XDG_CACHE_HOME": "cache/xdg",
                           "CUDA_CACHE_PATH": "cache/cuda", "TORCH_HOME": "cache/torch"}.items():
        path = run_root / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    command = job_command(args.python, args.variant, args.updates, output)
    record = dict(variant=args.variant, state="starting", pid=os.getpid(), command=command,
                  started_at=datetime.now(timezone.utc).isoformat(), console_log=str(log),
                  result_root=str(output), returncode=None)
    save_status(status, record)
    process = None
    code = 1
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1)
            record.update(state="running", child_pid=process.pid)
            save_status(status, record)
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                stream.write(line); stream.flush()
            code = process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        record.update(state="completed" if code == 0 else "failed", returncode=code,
                      ended_at=datetime.now(timezone.utc).isoformat())
        save_status(status, record)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
