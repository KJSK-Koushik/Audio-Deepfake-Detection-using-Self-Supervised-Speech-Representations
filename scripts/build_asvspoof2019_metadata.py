"""Create portable metadata CSV files from official ASVspoof 2019 LA protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.asvspoof2019 import write_metadata
from src.settings import ROOT_DIR, load_config


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT_DIR / config["data"]["raw_dir"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / config["data"]["metadata_dir"],
    )
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Generate metadata from protocols before the FLAC archive is fully extracted.",
    )
    args = parser.parse_args()

    summaries = write_metadata(
        args.dataset_root,
        args.output_dir,
        verify_audio=not args.skip_audio_check,
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
