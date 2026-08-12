from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"

REQUIRED_SECTIONS = {"project", "data", "audio", "baseline", "model", "training", "evaluation"}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config)
    return config


def validate_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    missing_sections = REQUIRED_SECTIONS - config.keys()
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"Missing configuration sections: {missing}")

    if config["audio"]["sample_rate"] <= 0:
        raise ValueError("audio.sample_rate must be positive.")

    if config["audio"]["max_duration_seconds"] <= 0:
        raise ValueError("audio.max_duration_seconds must be positive.")

    if config["audio"].get("crop_mode") not in {"start", "center"}:
        raise ValueError("audio.crop_mode must be 'start' or 'center'.")

    if config["data"]["labels"] != {"bonafide": 0, "spoof": 1}:
        raise ValueError("Labels must map bonafide to 0 and spoof to 1.")
