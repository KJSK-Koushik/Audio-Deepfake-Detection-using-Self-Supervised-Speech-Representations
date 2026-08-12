# Phase 2 Audio Preprocessing Report

## Objective

Phase 2 converts ASVspoof audio into a consistent input format without changing the official labels or split membership. The loader reads embedded FLAC bytes directly from the downloaded Parquet files.

## Preprocessing Contract

| Property | Decision |
|---|---|
| Output sample rate | 16,000 Hz |
| Output channels | Mono |
| Output duration | 6 seconds (96,000 samples) |
| Short recordings | Zero-pad at the end |
| Long recordings | Deterministic center crop |
| Padding indicator | Attention mask: `1` for audio, `0` for padding |
| Labels | Unchanged: bonafide `0`, spoof `1` |

Center cropping is deterministic, so repeated experiments receive the same input. Future training-only random cropping or augmentation can be evaluated separately without changing development data.

## Full Local Audit

Every downloaded train and development recording was decoded successfully.

| Split | Checked | Invalid | Sample rate | Channels | Minimum duration | Mean duration | Maximum duration |
|---|---:|---:|---|---|---:|---:|---:|
| Train | 25,380 | 0 | 16 kHz | Mono | 0.652 s | 3.426 s | 13.188 s |
| Development | 24,844 | 0 | 16 kHz | Mono | 0.695 s | 3.478 s | 11.594 s |
| Total | 50,224 | 0 | 16 kHz | Mono | 0.652 s | 3.452 s | 13.188 s |

The audit verifies decoding, finite samples, non-silent audio, label counts, sample rates, channel counts, durations, and peak amplitudes. Evaluation audio is not audited because it is intentionally deferred until the evaluation phase.

At the 6-second model boundary, 47,463 recordings require padding and 2,761 require center cropping. No recording is exactly 6 seconds long.

## Outputs

The following generated local files are ignored by Git:

- `data/processed/audio_audit.json`
- `data/processed/train_audio_manifest.csv` (25,380 rows)
- `data/processed/dev_audio_manifest.csv` (24,844 rows)

The manifests store Parquet file and row references plus audio properties. Fixed-length waveforms are produced on demand, avoiding a much larger duplicate audio dataset.

## Model Interface

`ASVspoofParquetDataset[index]` returns:

- `waveform`: float32 NumPy array with 96,000 samples
- `attention_mask`: 96,000 values identifying real audio and padding
- `sample_rate`: 16,000
- original sample rate, duration information, speaker, utterance, attack, and label fields

This common interface can feed Phase 3 MFCC/mel features and the later WavLM model.

## Approval Gate

Phase 3 has not started. It will build a reproducible MFCC or mel-spectrogram baseline only after explicit approval.
