from pathlib import Path

import pytest

from src.data.asvspoof2019 import (
    EXPECTED_COUNTS,
    count_missing_audio,
    parse_protocol,
    summarize_records,
    validate_official_counts,
    write_metadata,
)


def write_protocol(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_protocol_and_summary(tmp_path: Path) -> None:
    protocol = write_protocol(
        tmp_path / "train.txt",
        "LA_0079 LA_T_1000137 - - bonafide\n"
        "LA_0079 LA_T_1000406 - A01 spoof\n",
    )

    records = parse_protocol(protocol, "train")
    summary = summarize_records(records)

    assert records[0].label_id == 0
    assert records[1].audio_path == "ASVspoof2019_LA_train/flac/LA_T_1000406.flac"
    assert summary == {
        "total": 2,
        "bonafide": 1,
        "spoof": 1,
        "speakers": 1,
        "bonafide_speakers": 1,
        "attack_ids": ["A01"],
    }
    assert count_missing_audio(records, tmp_path) == 2


def test_rejects_malformed_protocol_row(tmp_path: Path) -> None:
    protocol = write_protocol(tmp_path / "bad.txt", "LA_0079 too-few-fields spoof\n")

    with pytest.raises(ValueError, match="must contain 5 fields"):
        parse_protocol(protocol, "train")


def test_rejects_inconsistent_attack_label(tmp_path: Path) -> None:
    protocol = write_protocol(tmp_path / "bad.txt", "LA_0079 LA_T_1 - - spoof\n")

    with pytest.raises(ValueError, match="must have an attack ID"):
        parse_protocol(protocol, "train")


def test_official_count_guard(tmp_path: Path) -> None:
    protocol = write_protocol(tmp_path / "train.txt", "LA_0079 LA_T_1 - - bonafide\n")
    records = parse_protocol(protocol, "train")

    with pytest.raises(ValueError, match="Unexpected train counts"):
        validate_official_counts(records, "train")

    assert EXPECTED_COUNTS["train"]["total"] == 25_380


def create_tiny_layout(tmp_path: Path, dev_speaker: str = "LA_0002") -> Path:
    protocol_dir = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True)
    write_protocol(
        protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        "LA_0001 LA_T_1 - - bonafide\n",
    )
    write_protocol(
        protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
        f"{dev_speaker} LA_D_1 - A01 spoof\n",
    )
    write_protocol(
        protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt",
        "LA_0003 LA_E_1 - - bonafide\n",
    )
    return tmp_path


def test_write_metadata_for_disjoint_splits(tmp_path: Path) -> None:
    dataset_root = create_tiny_layout(tmp_path)
    output_dir = tmp_path / "metadata"

    summaries = write_metadata(
        dataset_root,
        output_dir,
        verify_audio=False,
        strict_counts=False,
    )

    assert summaries["train"]["total"] == 1
    assert (output_dir / "train.csv").is_file()
    assert (output_dir / "summary.json").is_file()


def test_rejects_speaker_leakage_between_splits(tmp_path: Path) -> None:
    dataset_root = create_tiny_layout(tmp_path, dev_speaker="LA_0001")

    with pytest.raises(ValueError, match="Speaker appears in multiple splits"):
        write_metadata(
            dataset_root,
            tmp_path / "metadata",
            verify_audio=False,
            strict_counts=False,
        )
