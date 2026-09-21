"""Cross-platform ASVspoof Parquet downloader for local and Colab runs."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.settings import ROOT_DIR, load_config

REPOSITORY = "Bisher/ASVspoof_2019_LA"
REVISION = "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
SHARDS = {
    "train": {
        "filename": "data/train-00000-of-00001.parquet",
        "size": 1_586_086_073,
        "sha256": "b4eea1063bbcfa0c1cef1b69a96ad8b787c32f662005562b899cd4b461739619",
    },
    "dev": {
        "filename": "data/validation-00000-of-00001.parquet",
        "size": 1_575_535_827,
        "sha256": "9d2c340bb7f04c2d63ac018b224ecdb8c1855c789247232608b650f86c189b16",
    },
    "eval": {
        "filename": "data/test-00000-of-00001.parquet",
        "size": 4_381_501_164,
        "sha256": "8159935a94426bc36308278b23cecd8c4ca2a15b778a1087a4a0441d079c86af",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, split: str) -> bool:
    expected = SHARDS[split]
    return (
        path.is_file()
        and path.stat().st_size == expected["size"]
        and sha256(path) == expected["sha256"]
    )


def download_split(split: str, destination_root: Path) -> Path:
    destination = destination_root / f"{split}.parquet"
    if verify(destination, split):
        print(f"{split} is already complete and verified: {destination}")
        return destination

    staging = destination_root / ".download"
    downloaded = Path(
        hf_hub_download(
            repo_id=REPOSITORY,
            filename=str(SHARDS[split]["filename"]),
            repo_type="dataset",
            revision=REVISION,
            local_dir=staging,
            local_dir_use_symlinks=False,
        )
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.move(downloaded, destination)
    if not verify(destination, split):
        destination.unlink(missing_ok=True)
        raise ValueError(f"{split} download failed its size or SHA-256 check")
    print(f"{split} downloaded and verified: {destination}")
    return destination


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SHARDS),
        default=("train", "dev"),
    )
    args = parser.parse_args()
    destination_root = ROOT_DIR / config["data"]["parquet_dir"]
    for split in args.splits:
        download_split(split, destination_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
