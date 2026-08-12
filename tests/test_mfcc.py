import numpy as np
import pytest

from src.features.mfcc import MFCCConfig, extract_mfcc_statistics, feature_names


def test_mfcc_statistics_are_fixed_and_finite() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    features = extract_mfcc_statistics(waveform)

    assert features.shape == (240,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert len(feature_names()) == 240
    np.testing.assert_array_equal(features, extract_mfcc_statistics(waveform))


def test_mfcc_statistics_respect_coefficient_count() -> None:
    waveform = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)

    features = extract_mfcc_statistics(waveform, MFCCConfig(n_mfcc=20))

    assert features.shape == (120,)


def test_mfcc_rejects_empty_waveform() -> None:
    with pytest.raises(ValueError, match="non-empty mono"):
        extract_mfcc_statistics(np.array([], dtype=np.float32))
