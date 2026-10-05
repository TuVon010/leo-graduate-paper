import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from leo_routing.config import load_config
from leo_routing.topology.topology_cache import generate_topology
from leo_routing.utils.io import save_json


def main():
    parser = argparse.ArgumentParser(description="Generate and validate an offline topology trace")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/topology/base.npz")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    topology = generate_topology(config)
    topology.save(args.output)
    diagnostics = topology.diagnostics(config["simulation"]["slots"])
    save_json(args.output.with_suffix(".json"), diagnostics)
    print(diagnostics)
    print("Cache:", args.output.resolve())


if __name__ == "__main__":
    main()
