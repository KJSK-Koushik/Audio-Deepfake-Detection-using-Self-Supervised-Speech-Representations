# Project Specification

## Final Topic

Audio Deepfake Detection using Self-Supervised Speech Representations

## Problem

Given a speech recording, predict one of two labels:

- `bonafide` (`0`): genuine human speech
- `spoof` (`1`): synthesized or voice-converted speech

The system will learn from speech audio rather than transcripts. This keeps the project directly inside speech processing.

## Research Direction

The main model will use representations from a pretrained speech model such as WavLM or wav2vec 2.0. A small classifier will learn which representation patterns separate genuine and spoofed speech.

A classical MFCC or mel-spectrogram model will be built first as a baseline. The deep model must be compared with this baseline; a single accuracy value is not a complete experiment.

## Data

- Primary dataset: ASVspoof 2019 Logical Access (LA)
- Official protocol labels and train/development/evaluation splits will be preserved.
- No synthetic labels are required because the dataset already supplies ground truth.
- Noise, compression, gain, or speed perturbation may be introduced later as audio augmentation. Augmented samples inherit the original label.
- Optional generalization data: ASVspoof 2021 LA/DF, introduced only after the primary system works.

Dataset audio and generated metadata are stored locally under `data/` and are not committed to GitHub.

## Evaluation

The required metrics are accuracy, precision, recall, F1, ROC-AUC, equal error rate (EER), and a confusion matrix. The official dataset splits prevent the same speaker or sample from leaking between training and evaluation through a random resplit.

## Deliverables

- Reproducible preprocessing and dataset indexing code
- Classical baseline and SSL-based detector
- Saved model checkpoint and experiment configuration
- Metric tables, plots, and comparison/ablation results
- Streamlit demonstration for uploaded audio
- Final report, slides, architecture diagram, and viva notes

## Scope Boundaries

The first version is an offline binary detector for short speech recordings. Real-time streaming, source-generator identification, multilingual guarantees, and production security claims are outside the core scope. These may be explored only after the required system is complete.

## Main Risks and Controls

| Risk | Control |
|---|---|
| Deep training is slow without an NVIDIA GPU | Develop on CPU; train later on Colab, Kaggle, or another CUDA machine |
| Model memorizes a dataset rather than fake-speech cues | Preserve official splits and add optional cross-dataset evaluation |
| Accuracy hides class-specific errors | Report F1, ROC-AUC, EER, and confusion matrix |
| Large data/checkpoints overload Git | Keep them ignored and document how to reproduce them |
| Confidence is mistaken for certainty | Present it as a model score and tune the decision threshold on development data |

## Phase 0 Completion Criteria

- Repository structure and configuration exist.
- Local setup and environment diagnostics are reproducible.
- The Windows virtual environment uses a short path so PyTorch installs reliably.
- Configuration has automated smoke tests.
- GitHub CI runs linting and tests successfully.
- Project scope, data strategy, metrics, and phase gates are documented.
