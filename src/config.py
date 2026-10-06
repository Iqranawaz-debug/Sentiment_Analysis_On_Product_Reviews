"""Validated repository-relative experiment configuration."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "configs/default.yaml"):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["split"] != {"train": 0.70, "val": 0.15, "test": 0.15}:
        raise ValueError("The evaluation protocol requires a 70/15/15 split.")
    if config["cv_folds"] != 5:
        raise ValueError("The evaluation protocol requires five CV folds.")
    if not 0 < config["calibration_fraction"] < 1:
        raise ValueError("calibration_fraction must be between zero and one.")
    if not 0 <= config["low_confidence_threshold"] <= 1:
        raise ValueError("Invalid confidence threshold.")
    for key in ("models", "reports"):
        config["paths"][key] = str((ROOT / config["paths"][key]).resolve())
    config["paths"]["data"] = [str((ROOT / p).resolve()) for p in config["paths"]["data"]]
    return config
