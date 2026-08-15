"""WavLM data utilities shared by training and inference."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Subset

from src.data.audio import ASVspoofParquetDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_parquet_labels(parquet_path: str | Path) -> np.ndarray:
    labels = pq.read_table(parquet_path, columns=["key"])["key"].to_numpy()
    return np.asarray(labels, dtype=np.int64)


def stratified_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Choose a deterministic proportional subset that includes every class."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= len(labels):
        return np.arange(len(labels), dtype=np.int64)

    classes, counts = np.unique(labels, return_counts=True)
    if limit < len(classes):
        raise ValueError("limit must be at least the number of classes")

    rng = np.random.default_rng(seed)
    allocations = np.maximum(1, np.floor(limit * counts / len(labels)).astype(int))
    while allocations.sum() < limit:
        allocations[np.argmax(counts - allocations)] += 1
    while allocations.sum() > limit:
        candidates = np.where(allocations > 1)[0]
        allocations[candidates[np.argmax(allocations[candidates])]] -= 1

    selected = []
    for class_id, allocation in zip(classes, allocations, strict=True):
        class_indices = np.flatnonzero(labels == class_id)
        selected.extend(rng.choice(class_indices, size=allocation, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def make_audio_dataset(
    parquet_path: str | Path,
    audio_config: dict[str, object],
    *,
    limit: int | None = None,
    seed: int = 42,
) -> ASVspoofParquetDataset | Subset:
    dataset = ASVspoofParquetDataset(
        parquet_path,
        target_rate=int(audio_config["sample_rate"]),
        max_duration_seconds=float(audio_config["max_duration_seconds"]),
        crop_mode=str(audio_config["crop_mode"]),
        pad_value=float(audio_config["pad_value"]),
    )
    if limit is None or limit >= len(dataset):
        return dataset
    labels = load_parquet_labels(parquet_path)
    return Subset(dataset, stratified_indices(labels, limit, seed).tolist())


def collate_audio_records(records: Sequence[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("Cannot collate an empty batch")
    return {
        "input_values": torch.from_numpy(
            np.stack([record["waveform"] for record in records])
        ).float(),
        "attention_mask": torch.from_numpy(
            np.stack([record["attention_mask"] for record in records])
        ).long(),
        "labels": torch.tensor([record["label_id"] for record in records], dtype=torch.long),
        "utterance_ids": [str(record["utterance_id"]) for record in records],
    }


def balanced_class_weights(labels: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("Both label classes 0 and 1 are required")
    weights = len(labels) / (len(classes) * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)
