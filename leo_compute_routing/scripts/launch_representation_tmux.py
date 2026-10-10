"""Launch matched, scratch-trained representation experiments in tmux windows."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "mlp", "self_graph", "mlp_kkt", "gat_kkt", "gated_kkt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["self_graph", "mlp_kkt", "gat_kkt", "gated_kkt"])
    parser.add_argument("--reference-results", type=Path,
                        help="Optional completed base MLP/GAT results, for matched-input summary only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.updates < 1 or len(set(args.variants)) != len(args.variants):
        parser.error("Use positive updates and distinct variants")
    if not args.session or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.session):
        parser.error("Use a simple tmux session name")
    root = args.run_root.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Run root is nonempty; choose a new timestamp")
    windows = {v: [sys.executable, "-u", str(ROOT / "scripts/run_representation_window.py"),
                   "--run-root", str(root), "--variant", v, "--updates", str(args.updates),
                   "--python", sys.executable] for v in args.variants}
    if args.dry_run:
        print(json.dumps(windows, indent=2)); return
    if shutil.which("tmux") is None:
        parser.error("tmux is required")
    if subprocess.run(["tmux", "has-session", "-t", args.session], capture_output=True).returncode == 0:
        parser.error("tmux session exists; choose another name")
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 10 * 1024 ** 3:
        parser.error("Run filesystem has less than 10 GiB free")
    probe = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"],
                           capture_output=True, text=True)
    if probe.returncode:
        parser.error(probe.stderr.strip())
    root.mkdir(parents=True)
    def git(*parts):
        result = subprocess.run(["git", *parts], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    manifest = dict(schema=1, session=args.session, repository=str(ROOT), run_root=str(root),
                    created_at=datetime.now(timezone.utc).isoformat(), branch=git("branch", "--show-current"),
                    commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
                    variants=args.variants, updates=args.updates, training_root_seed=2026,
                    validation_seed=100, randomized_episodes=True, initialization="scratch",
                    shared_physics="configs/experiments/coupled24.yaml", training_device="cuda",
                    reference_results=str(args.reference_results.resolve()) if args.reference_results else None,
                    evaluation_role="validation_reuse; not independent test", windows=windows)
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    monitor = [sys.executable, "-u", str(ROOT / "scripts/representation_status.py"), str(root), "--watch"]
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session, "-n", "monitor", "-c", str(ROOT),
                    shlex.join(monitor)], check=True)
    for variant, command in windows.items():
        subprocess.run(["tmux", "new-window", "-d", "-t", args.session, "-n", variant, "-c", str(ROOT),
                        shlex.join(command)], check=True)
        subprocess.run(["tmux", "set-window-option", "-t", args.session + ":" + variant,
                        "remain-on-exit", "on"], check=True, stdout=subprocess.DEVNULL)
    print("[LAUNCHED] session=%s variants=%s root=%s" % (args.session, ",".join(args.variants), root))
    print("[ATTACH] tmux attach -t " + args.session)
    print("[STATUS] " + shlex.join([sys.executable, str(ROOT / "scripts/representation_status.py"), str(root)]))


if __name__ == "__main__":
    main()
