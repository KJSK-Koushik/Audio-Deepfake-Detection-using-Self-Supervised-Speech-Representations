"""ASVspoof 2019 Logical Access protocol parsing and metadata generation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

LABEL_IDS = {"bonafide": 0, "spoof": 1}
SPLIT_DIRECTORIES = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}
PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
EXPECTED_COUNTS = {
    "train": {"total": 25_380, "bonafide": 2_580, "spoof": 22_800},
    "dev": {"total": 24_844, "bonafide": 2_548, "spoof": 22_296},
    "eval": {"total": 71_237, "bonafide": 7_355, "spoof": 63_882},
}

CSV_FIELDS = [
    "speaker_id",
    "utterance_id",
    "attack_id",
    "label",
    "label_id",
    "split",
    "audio_path",
]


@dataclass(frozen=True)
class ProtocolRecord:
    speaker_id: str
    utterance_id: str
    attack_id: str
    label: str
    label_id: int
    split: str
    audio_path: str


def parse_protocol(protocol_path: Path, split: str) -> list[ProtocolRecord]:
    if split not in SPLIT_DIRECTORIES:
        raise ValueError(f"Unknown split: {split}")

    records = []
    seen_utterances: set[str] = set()

    with protocol_path.open("r", encoding="utf-8") as protocol_file:
        for line_number, raw_line in enumerate(protocol_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{protocol_path}:{line_number} must contain 5 fields, found {len(fields)}"
                )

            speaker_id, utterance_id, _unused, attack_id, label = fields
            if label not in LABEL_IDS:
                raise ValueError(f"{protocol_path}:{line_number} has unknown label: {label}")
            if utterance_id in seen_utterances:
                raise ValueError(f"Duplicate utterance ID in {split}: {utterance_id}")
            if label == "bonafide" and attack_id != "-":
                raise ValueError(f"Bonafide utterance {utterance_id} must use attack ID '-'")
            if label == "spoof" and attack_id == "-":
                raise ValueError(f"Spoof utterance {utterance_id} must have an attack ID")

            seen_utterances.add(utterance_id)
            relative_audio_path = (
                Path(SPLIT_DIRECTORIES[split]) / "flac" / f"{utterance_id}.flac"
            )
            records.append(
                ProtocolRecord(
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    attack_id=attack_id,
                    label=label,
                    label_id=LABEL_IDS[label],
                    split=split,
                    audio_path=relative_audio_path.as_posix(),
                )
            )

    if not records:
        raise ValueError(f"Protocol is empty: {protocol_path}")

    return records


def summarize_records(records: Iterable[ProtocolRecord]) -> dict[str, object]:
    records = list(records)
    labels = Counter(record.label for record in records)
    speakers = {record.speaker_id for record in records}
    bonafide_speakers = {
        record.speaker_id for record in records if record.label == "bonafide"
    }
    attacks = {record.attack_id for record in records if record.attack_id != "-"}
    return {
        "total": len(records),
        "bonafide": labels["bonafide"],
        "spoof": labels["spoof"],
        "speakers": len(speakers),
        "bonafide_speakers": len(bonafide_speakers),
        "attack_ids": sorted(attacks),
    }


def validate_official_counts(records: Iterable[ProtocolRecord], split: str) -> None:
    summary = summarize_records(records)
    expected = EXPECTED_COUNTS[split]
    actual_counts = {key: summary[key] for key in expected}
    if actual_counts != expected:
        raise ValueError(f"Unexpected {split} counts: expected {expected}, found {actual_counts}")

def count_missing_audio(records: Iterable[ProtocolRecord], dataset_root: Path) -> int:
    return sum(not (dataset_root / record.audio_path).is_file() for record in records)


def find_dataset_layout(dataset_root: Path) -> tuple[Path, Path]:
    protocol_name = PROTOCOL_FILES["train"]
    matches = list(dataset_root.rglob(protocol_name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {protocol_name} below {dataset_root}. Extract the official LA.zip first."
        )
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise ValueError(f"Multiple ASVspoof protocol layouts found: {paths}")

    protocol_dir = matches[0].parent
    layout_root = protocol_dir.parent
    return layout_root, protocol_dir


def write_metadata(
    dataset_root: Path,
    output_dir: Path,
    *,
    verify_audio: bool = True,
    strict_counts: bool = True,
) -> dict[str, dict[str, object]]:
    layout_root, protocol_dir = find_dataset_layout(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, dict[str, object]] = {}
    seen_utterances: set[str] = set()
    seen_speakers: set[str] = set()

    for split, protocol_filename in PROTOCOL_FILES.items():
        protocol_path = protocol_dir / protocol_filename
        if not protocol_path.is_file():
            raise FileNotFoundError(f"Missing protocol file: {protocol_path}")

        records = parse_protocol(protocol_path, split)
        if strict_counts:
            validate_official_counts(records, split)

        split_utterances = {record.utterance_id for record in records}
        duplicate_utterances = seen_utterances & split_utterances
        if duplicate_utterances:
            duplicate = min(duplicate_utterances)
            raise ValueError(f"Utterance appears in multiple splits: {duplicate}")

        split_speakers = {record.speaker_id for record in records}
        duplicate_speakers = seen_speakers & split_speakers
        if duplicate_speakers:
            duplicate = min(duplicate_speakers)
            raise ValueError(f"Speaker appears in multiple splits: {duplicate}")

        seen_utterances.update(split_utterances)
        seen_speakers.update(split_speakers)

        missing_audio = count_missing_audio(records, layout_root) if verify_audio else None
        if verify_audio and missing_audio:
            raise FileNotFoundError(f"{split} is missing {missing_audio} referenced FLAC files")

        csv_path = output_dir / f"{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)

        summary = summarize_records(records)
        summary["missing_audio"] = missing_audio
        summary["metadata_file"] = csv_path.as_posix()
        all_summaries[split] = summary

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(all_summaries, summary_file, indent=2)
        summary_file.write("\n")

    return all_summaries
