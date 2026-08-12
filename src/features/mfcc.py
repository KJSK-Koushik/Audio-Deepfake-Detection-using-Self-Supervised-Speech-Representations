"""MFCC statistics for the classical audio deepfake baseline."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class MFCCConfig:
    sample_rate: int = 16_000
    n_mfcc: int = 40
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    fmin: float = 20.0
    fmax: float = 7_600.0


DEFAULT_MFCC_CONFIG = MFCCConfig()


def extract_mfcc_statistics(
    waveform: np.ndarray,
    config: MFCCConfig = DEFAULT_MFCC_CONFIG,
) -> np.ndarray:
    """Extract mean/std summaries of MFCC, delta, and delta-delta frames."""
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform must be a non-empty mono array")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or infinite values")

    mfcc = librosa.feature.mfcc(
        y=np.asarray(waveform, dtype=np.float32),
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.n_fft,
        window="hann",
        center=False,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    if mfcc.shape[1] < 9:
        raise ValueError("waveform is too short for delta features")

    delta = librosa.feature.delta(mfcc, order=1, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
    features = np.concatenate((mfcc, delta, delta2), axis=0)
    summary = np.concatenate((features.mean(axis=1), features.std(axis=1)))
    return np.ascontiguousarray(summary, dtype=np.float32)


def feature_names(n_mfcc: int = 40) -> list[str]:
    names = []
    for statistic in ("mean", "std"):
        for family in ("mfcc", "delta", "delta2"):
            names.extend(f"{family}_{index:02d}_{statistic}" for index in range(n_mfcc))
    return names
