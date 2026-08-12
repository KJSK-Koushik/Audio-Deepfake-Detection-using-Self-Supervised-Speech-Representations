"""Audio decoding and indexed access for ASVspoof Parquet files."""

from __future__ import annotations

import io
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray
    sample_rate: int
    channels: int


def decode_audio_bytes(audio_bytes: bytes) -> DecodedAudio:
    """Decode trusted audio bytes and mix multiple channels to mono."""
    if not audio_bytes:
        raise ValueError("Audio payload is empty")

    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True,
        )
    except sf.LibsndfileError as error:
        raise ValueError("Audio payload cannot be decoded") from error

    if sample_rate <= 0 or waveform.shape[0] == 0:
        raise ValueError("Decoded audio has no samples or an invalid sample rate")
    if not np.isfinite(waveform).all():
        raise ValueError("Decoded audio contains NaN or infinite values")

    channels = waveform.shape[1]
    mono = waveform.mean(axis=1, dtype=np.float32)
    return DecodedAudio(np.ascontiguousarray(mono), sample_rate, channels)


def resample_audio(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample mono audio while preserving its expected time duration."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono (one-dimensional)")
    if source_rate == target_rate:
        return np.ascontiguousarray(waveform, dtype=np.float32)

    divisor = math.gcd(source_rate, target_rate)
    resampled = resample_poly(
        waveform,
        up=target_rate // divisor,
        down=source_rate // divisor,
    ).astype(np.float32, copy=False)

    expected_samples = round(len(waveform) * target_rate / source_rate)
    if len(resampled) > expected_samples:
        resampled = resampled[:expected_samples]
    elif len(resampled) < expected_samples:
        resampled = np.pad(resampled, (0, expected_samples - len(resampled)))
    return np.ascontiguousarray(resampled)


def crop_or_pad(
    waveform: np.ndarray,
    target_samples: int,
    *,
    crop_mode: str = "center",
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-length audio and a mask that distinguishes audio from padding."""
    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono (one-dimensional)")
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if crop_mode not in {"start", "center"}:
        raise ValueError("crop_mode must be 'start' or 'center'")

    if len(waveform) >= target_samples:
        start = 0 if crop_mode == "start" else (len(waveform) - target_samples) // 2
        output = waveform[start : start + target_samples]
        mask = np.ones(target_samples, dtype=np.int8)
        return np.ascontiguousarray(output, dtype=np.float32), mask

    padding = target_samples - len(waveform)
    output = np.pad(waveform, (0, padding), constant_values=pad_value).astype(
        np.float32, copy=False
    )
    mask = np.concatenate(
        (np.ones(len(waveform), dtype=np.int8), np.zeros(padding, dtype=np.int8))
    )
    return np.ascontiguousarray(output), mask


def prepare_audio(
    decoded: DecodedAudio,
    *,
    target_rate: int = 16_000,
    max_duration_seconds: float = 6.0,
    crop_mode: str = "center",
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample and convert decoded audio into fixed-length model input."""
    target_samples = round(target_rate * max_duration_seconds)
    resampled = resample_audio(decoded.waveform, decoded.sample_rate, target_rate)
    return crop_or_pad(
        resampled,
        target_samples,
        crop_mode=crop_mode,
        pad_value=pad_value,
    )


class ASVspoofParquetDataset:
    """Random-access train/dev dataset backed by 100-row Parquet groups."""

    REQUIRED_COLUMNS = {
        "speaker_id",
        "audio_file_name",
        "audio",
        "system_id",
        "key",
    }

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        target_rate: int = 16_000,
        max_duration_seconds: float = 6.0,
        crop_mode: str = "center",
        pad_value: float = 0.0,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"Parquet dataset not found: {self.parquet_path}")

        self.target_rate = target_rate
        self.max_duration_seconds = max_duration_seconds
        self.crop_mode = crop_mode
        self.pad_value = pad_value
        self._parquet = pq.ParquetFile(self.parquet_path)

        columns = set(self._parquet.schema_arrow.names)
        missing = self.REQUIRED_COLUMNS - columns
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Parquet dataset is missing columns: {names}")

        self._row_group_starts = [0]
        for group_index in range(self._parquet.num_row_groups):
            group_rows = self._parquet.metadata.row_group(group_index).num_rows
            self._row_group_starts.append(self._row_group_starts[-1] + group_rows)

        self._cached_group_index: int | None = None
        self._cached_group: list[dict[str, object]] = []

    def __len__(self) -> int:
        return self._row_group_starts[-1]

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_parquet"] = None
        state["_cached_group_index"] = None
        state["_cached_group"] = []
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._parquet = pq.ParquetFile(self.parquet_path)

    def _read_record(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        group_index = bisect_right(self._row_group_starts, index) - 1
        if group_index != self._cached_group_index:
            self._cached_group = self._parquet.read_row_group(group_index).to_pylist()
            self._cached_group_index = group_index

        local_index = index - self._row_group_starts[group_index]
        return self._cached_group[local_index]

    def raw_record(self, index: int) -> dict[str, object]:
        """Return metadata plus decoded, unresampled mono audio."""
        record = self._read_record(index)
        audio = record["audio"]
        if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
            raise ValueError(f"Row {index} does not contain embedded audio bytes")

        decoded = decode_audio_bytes(audio["bytes"])
        return {
            "waveform": decoded.waveform,
            "sample_rate": decoded.sample_rate,
            "channels": decoded.channels,
            "speaker_id": str(record["speaker_id"]),
            "utterance_id": str(record["audio_file_name"]),
            "attack_id": str(record["system_id"]),
            "label_id": int(record["key"]),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.raw_record(index)
        decoded = DecodedAudio(
            waveform=record["waveform"],
            sample_rate=record["sample_rate"],
            channels=record["channels"],
        )
        waveform, attention_mask = prepare_audio(
            decoded,
            target_rate=self.target_rate,
            max_duration_seconds=self.max_duration_seconds,
            crop_mode=self.crop_mode,
            pad_value=self.pad_value,
        )
        return {
            **record,
            "waveform": waveform,
            "attention_mask": attention_mask,
            "sample_rate": self.target_rate,
            "original_sample_rate": decoded.sample_rate,
            "original_num_samples": len(decoded.waveform),
        }
