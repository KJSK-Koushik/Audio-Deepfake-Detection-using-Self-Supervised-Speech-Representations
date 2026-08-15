import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.models.wavlm import (  # noqa: E402
    balanced_class_weights,
    collate_audio_records,
    stratified_indices,
)


def test_stratified_indices_are_reproducible_and_include_both_classes() -> None:
    labels = np.array([0] * 10 + [1] * 90)

    first = stratified_indices(labels, limit=20, seed=42)
    second = stratified_indices(labels, limit=20, seed=42)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 20
    assert set(labels[first]) == {0, 1}


def test_balanced_class_weights_upweight_minority() -> None:
    weights = balanced_class_weights(np.array([0, 1, 1, 1]))

    assert weights.shape == (2,)
    assert weights[0] > weights[1]


def test_collate_audio_records_returns_wavlm_tensors() -> None:
    records = [
        {
            "waveform": np.zeros(16, dtype=np.float32),
            "attention_mask": np.ones(16, dtype=np.int8),
            "label_id": 0,
            "utterance_id": "LA_T_1",
        },
        {
            "waveform": np.ones(16, dtype=np.float32),
            "attention_mask": np.ones(16, dtype=np.int8),
            "label_id": 1,
            "utterance_id": "LA_T_2",
        },
    ]

    batch = collate_audio_records(records)

    assert batch["input_values"].shape == (2, 16)
    assert batch["attention_mask"].dtype == torch.int64
    assert batch["labels"].tolist() == [0, 1]
