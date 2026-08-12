"""Audit downloaded ASVspoof audio and create model-ready local manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.audio import ASVspoofParquetDataset
from src.settings import ROOT_DIR, load_config

MANIFEST_FIELDS = [
    "parquet_file",
    "parquet_row_index",
    "utterance_id",
    "speaker_id",
    "attack_id",
    "label_id",
    "source_sample_rate",
    "source_channels",
    "source_num_samples",
    "duration_seconds",
    "peak_amplitude",
]


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def audit_split(
    split: str,
    parquet_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    max_duration_seconds: float = 6.0,
) -> dict[str, object]:
    dataset = ASVspoofParquetDataset(parquet_path)
    rows_to_check = len(dataset) if limit is None else min(limit, len(dataset))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split}_audio_manifest.csv"
    manifest_parquet_path = portable_path(parquet_path)

    sample_rates: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    durations: list[float] = []
    peaks: list[float] = []
    invalid_rows: list[dict[str, object]] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for index in range(rows_to_check):
            try:
                record = dataset.raw_record(index)
                duration = len(record["waveform"]) / record["sample_rate"]
                peak = float(np.max(np.abs(record["waveform"])))
                if peak <= 1e-7:
                    raise ValueError("Decoded audio is silent")
                writer.writerow(
                    {
                        "parquet_file": manifest_parquet_path,
                        "parquet_row_index": index,
                        "utterance_id": record["utterance_id"],
                        "speaker_id": record["speaker_id"],
                        "attack_id": record["attack_id"],
                        "label_id": record["label_id"],
                        "source_sample_rate": record["sample_rate"],
                        "source_channels": record["channels"],
                        "source_num_samples": len(record["waveform"]),
                        "duration_seconds": f"{duration:.6f}",
                        "peak_amplitude": f"{peak:.8f}",
                    }
                )
                sample_rates[record["sample_rate"]] += 1
                channel_counts[record["channels"]] += 1
                label_counts[record["label_id"]] += 1
                durations.append(duration)
                peaks.append(peak)
            except (ValueError, RuntimeError) as error:
                invalid_rows.append({"row_index": index, "error": str(error)})

    duration_array = np.asarray(durations, dtype=np.float64)
    peak_array = np.asarray(peaks, dtype=np.float64)
    return {
        "split": split,
        "parquet_file": manifest_parquet_path,
        "rows_in_file": len(dataset),
        "rows_checked": rows_to_check,
        "valid_audio": len(durations),
        "invalid_audio": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "sample_rates": dict(sorted(sample_rates.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "duration_seconds": {
            "minimum": float(duration_array.min()) if len(duration_array) else None,
            "mean": float(duration_array.mean()) if len(duration_array) else None,
            "maximum": float(duration_array.max()) if len(duration_array) else None,
        },
        "fixed_length_processing": {
            "target_seconds": max_duration_seconds,
            "will_be_padded": sum(duration < max_duration_seconds for duration in durations),
            "will_be_cropped": sum(duration > max_duration_seconds for duration in durations),
            "already_exact": sum(duration == max_duration_seconds for duration in durations),
        },
        "peak_amplitude": {
            "minimum": float(peak_array.min()) if len(peak_array) else None,
            "mean": float(peak_array.mean()) if len(peak_array) else None,
            "maximum": float(peak_array.max()) if len(peak_array) else None,
        },
        "manifest_file": portable_path(manifest_path),
    }


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=("train", "dev"))
    parser.add_argument("--limit", type=int, help="Audit only the first N rows per split")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    parquet_dir = ROOT_DIR / config["data"]["parquet_dir"]
    output_dir = ROOT_DIR / config["data"]["processed_dir"]
    results = {
        split: audit_split(
            split,
            parquet_dir / f"{split}.parquet",
            output_dir,
            limit=args.limit,
            max_duration_seconds=config["audio"]["max_duration_seconds"],
        )
        for split in args.splits
    }

    report_path = output_dir / "audio_audit.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    if any(result["invalid_audio"] for result in results.values()):
        return 1
    print(f"Audio audit written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
