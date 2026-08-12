"""Fetch and split checksum-pinned ASVspoof 2019 LA protocol metadata."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from contextlib import ExitStack
from pathlib import Path

from src.data.asvspoof2019 import EXPECTED_COUNTS, PROTOCOL_FILES
from src.settings import ROOT_DIR, load_config

SOURCE_URL = (
    "https://raw.githubusercontent.com/nii-yamagishilab/project-NN-Pytorch-scripts/"
    "112ec0ff021e58a6c92e945547e7bfb21c11424e/"
    "project/02-asvspoof/DATA/asvspoof2019_LA/protocol.txt"
)
SOURCE_SHA256 = "f5efc7385ea6ff810c0debf68cead3fc27200511f3e7d692057de08810018d38"
PREFIX_TO_SPLIT = {"LA_T_": "train", "LA_D_": "dev", "LA_E_": "eval"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_source(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "audio-deepfake-project/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        with destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)

    actual_hash = file_sha256(destination)
    if actual_hash != SOURCE_SHA256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Protocol checksum mismatch: expected {SOURCE_SHA256}, found {actual_hash}")


def split_source(source: Path, protocol_dir: Path) -> dict[str, int]:
    protocol_dir.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(PROTOCOL_FILES, 0)

    with ExitStack() as stack:
        outputs = {
            split: stack.enter_context(
                (protocol_dir / filename).open("w", encoding="utf-8", newline="\n")
            )
            for split, filename in PROTOCOL_FILES.items()
        }

        with source.open("r", encoding="utf-8") as source_file:
            for line_number, raw_line in enumerate(source_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Combined protocol line {line_number} is malformed")

                utterance_id = fields[1]
                split = next(
                    (value for prefix, value in PREFIX_TO_SPLIT.items() if utterance_id.startswith(prefix)),
                    None,
                )
                if split is None:
                    raise ValueError(f"Unknown utterance prefix at line {line_number}: {utterance_id}")

                outputs[split].write(line + "\n")
                counts[split] += 1

    expected_totals = {split: values["total"] for split, values in EXPECTED_COUNTS.items()}
    if counts != expected_totals:
        raise ValueError(f"Protocol split counts do not match ASVspoof: {counts}")
    return counts


def main() -> int:
    config = load_config()
    dataset_root = ROOT_DIR / config["data"]["raw_dir"]
    source_path = dataset_root / "protocol_combined.txt"
    protocol_dir = dataset_root / "LA" / "ASVspoof2019_LA_cm_protocols"

    if not source_path.is_file() or file_sha256(source_path) != SOURCE_SHA256:
        fetch_source(source_path)

    counts = split_source(source_path, protocol_dir)
    print(f"Verified source SHA-256: {SOURCE_SHA256}")
    for split, count in counts.items():
        print(f"{split:5}: {count:,} rows")
    print(f"Protocols written to: {protocol_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
