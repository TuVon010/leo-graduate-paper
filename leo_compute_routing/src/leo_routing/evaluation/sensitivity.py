from copy import deepcopy
from pathlib import Path

from ..config import validate_config
from ..utils.io import save_csv
from .evaluator import run_comparison


def run_sensitivity(config, parameter, values, seeds, algorithms, output_directory, progress=None):
    section, separator, field = parameter.partition(".")
    if not separator or section not in config or field not in config[section]:
        raise ValueError("Parameter must name an existing section.key")
    all_rows = []
    for index, value in enumerate(values):
        for seed in seeds:
            variant = deepcopy(config)
            variant[section][field] = value
            variant["simulation"]["seed"] = seed
            validate_config(variant)
            output = Path(output_directory) / ("value_%02d" % index) / ("seed_%s" % seed)
            rows = run_comparison(variant, algorithms, output, progress=progress)
            all_rows.extend({"parameter": parameter, "parameter_value": value, **row} for row in rows)
    save_csv(Path(output_directory) / "sensitivity.csv", all_rows)
    return all_rows
