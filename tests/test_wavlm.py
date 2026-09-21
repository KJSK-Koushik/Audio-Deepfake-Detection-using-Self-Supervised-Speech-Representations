import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.models.wavlm import (  # noqa: E402
    balanced_class_weights,
    collate_audio_records,
    configure_partial_finetuning,
    stratified_indices,
)


class DummyWavLMClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wavlm = torch.nn.Module()
        self.wavlm.feature_projection = torch.nn.Linear(2, 2)
        self.wavlm.encoder = torch.nn.Module()
        self.wavlm.encoder.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(4)]
        )
        self.wavlm.encoder.layer_norm = torch.nn.LayerNorm(2)
        self.classifier = torch.nn.Linear(2, 2)


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


def test_partial_finetuning_keeps_only_last_layers_and_head_trainable() -> None:
    model = DummyWavLMClassifier()

    counts = configure_partial_finetuning(model, trainable_transformer_layers=2)

    assert not any(parameter.requires_grad for parameter in model.wavlm.feature_projection.parameters())
    assert not any(parameter.requires_grad for parameter in model.wavlm.encoder.layers[0].parameters())
    assert not any(parameter.requires_grad for parameter in model.wavlm.encoder.layers[1].parameters())
    assert all(parameter.requires_grad for parameter in model.wavlm.encoder.layers[2].parameters())
    assert all(parameter.requires_grad for parameter in model.wavlm.encoder.layers[3].parameters())
    assert all(parameter.requires_grad for parameter in model.wavlm.encoder.layer_norm.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert counts["trainable_transformer_layers"] == 2


def test_partial_finetuning_rejects_too_many_layers() -> None:
    with pytest.raises(ValueError, match="between 0 and 4"):
        configure_partial_finetuning(DummyWavLMClassifier(), trainable_transformer_layers=5)
