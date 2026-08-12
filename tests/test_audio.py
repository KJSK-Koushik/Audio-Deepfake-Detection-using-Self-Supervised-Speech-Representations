import io
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from scripts.audit_asvspoof2019_audio import audit_split
from src.data.audio import (
    ASVspoofParquetDataset,
    crop_or_pad,
    decode_audio_bytes,
    prepare_audio,
)


def encoded_flac(waveform: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="FLAC")
    return buffer.getvalue()


def test_decode_stereo_audio_to_mono() -> None:
    stereo = np.column_stack(
        (np.full(800, 0.25, dtype=np.float32), np.full(800, -0.25, dtype=np.float32))
    )

    decoded = decode_audio_bytes(encoded_flac(stereo, 8_000))

    assert decoded.sample_rate == 8_000
    assert decoded.channels == 2
    assert decoded.waveform.shape == (800,)
    assert np.max(np.abs(decoded.waveform)) < 1e-4


def test_prepare_audio_resamples_and_pads_with_mask() -> None:
    source = np.sin(np.linspace(0, 20, 4_000, dtype=np.float32))
    decoded = decode_audio_bytes(encoded_flac(source, 8_000))

    waveform, mask = prepare_audio(
        decoded,
        target_rate=16_000,
        max_duration_seconds=1.0,
    )

    assert waveform.shape == (16_000,)
    assert mask.shape == (16_000,)
    assert mask.sum() == 8_000
    assert np.all(waveform[8_000:] == 0)


def test_center_crop_is_deterministic() -> None:
    source = np.arange(10, dtype=np.float32)

    first, first_mask = crop_or_pad(source, 4, crop_mode="center")
    second, second_mask = crop_or_pad(source, 4, crop_mode="center")

    np.testing.assert_array_equal(first, np.array([3, 4, 5, 6], dtype=np.float32))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_mask, second_mask)


def test_rejects_invalid_audio_bytes() -> None:
    with pytest.raises(ValueError, match="cannot be decoded"):
        decode_audio_bytes(b"not audio")


def test_parquet_dataset_returns_model_ready_record(tmp_path: Path) -> None:
    audio_bytes = encoded_flac(np.ones(800, dtype=np.float32) * 0.1, 8_000)
    table = pa.Table.from_pylist(
        [
            {
                "speaker_id": "LA_0001",
                "audio_file_name": "LA_T_0001",
                "audio": {"bytes": audio_bytes, "path": "LA_T_0001.flac"},
                "system_id": "-",
                "key": 0,
            }
        ]
    )
    parquet_path = tmp_path / "tiny.parquet"
    pq.write_table(table, parquet_path, row_group_size=1)

    dataset = ASVspoofParquetDataset(
        parquet_path,
        target_rate=16_000,
        max_duration_seconds=0.2,
    )
    record = dataset[0]

    assert len(dataset) == 1
    assert record["utterance_id"] == "LA_T_0001"
    assert record["label_id"] == 0
    assert record["waveform"].shape == (3_200,)
    assert record["attention_mask"].sum() == 1_600
    assert record["original_sample_rate"] == 8_000


def test_audio_audit_writes_manifest_and_summary(tmp_path: Path) -> None:
    audio_bytes = encoded_flac(np.ones(800, dtype=np.float32) * 0.1, 8_000)
    table = pa.Table.from_pylist(
        [
            {
                "speaker_id": "LA_0001",
                "audio_file_name": "LA_T_0001",
                "audio": {"bytes": audio_bytes, "path": "LA_T_0001.flac"},
                "system_id": "-",
                "key": 0,
            }
        ]
    )
    parquet_path = tmp_path / "tiny.parquet"
    pq.write_table(table, parquet_path)

    summary = audit_split("train", parquet_path, tmp_path / "processed")

    assert summary["valid_audio"] == 1
    assert summary["invalid_audio"] == 0
    assert summary["sample_rates"] == {8_000: 1}
    assert summary["fixed_length_processing"]["will_be_padded"] == 1
    assert (tmp_path / "processed" / "train_audio_manifest.csv").is_file()


def test_parquet_dataset_can_move_to_spawned_data_worker(tmp_path: Path) -> None:
    audio_bytes = encoded_flac(np.ones(800, dtype=np.float32) * 0.1, 8_000)
    table = pa.Table.from_pylist(
        [
            {
                "speaker_id": "LA_0001",
                "audio_file_name": "LA_T_0001",
                "audio": {"bytes": audio_bytes, "path": "LA_T_0001.flac"},
                "system_id": "-",
                "key": 0,
            }
        ]
    )
    parquet_path = tmp_path / "tiny.parquet"
    pq.write_table(table, parquet_path)

    restored = pickle.loads(pickle.dumps(ASVspoofParquetDataset(parquet_path)))

    assert len(restored) == 1
    assert restored[0]["utterance_id"] == "LA_T_0001"
