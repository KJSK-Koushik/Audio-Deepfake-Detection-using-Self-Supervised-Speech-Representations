# Audio Deepfake Detection using Self-Supervised Speech Representations

This project detects whether a speech audio sample is real human speech (`bonafide`) or fake/spoofed/generated speech (`spoof`).

## Project Scope

- Primary dataset: ASVspoof 2019 Logical Access (LA)
- Main model direction: wav2vec2/WavLM speech representations with a binary classifier
- Baseline direction: MFCC or mel-spectrogram features with a classical ML classifier
- Demo direction: Streamlit app for audio upload and real/fake prediction

## Why This Is In Scope

The task is speech-processing specific because the input is speech audio, and the system learns audio/speech representations before making a real/fake decision.

## Planned Phases

| Phase | What gets built | Output |
|---:|---|---|
| 0 | Project scaffold, environment, config, CI smoke test | Working skeleton |
| 1 | ASVspoof dataset setup and metadata indexing | Train/dev/eval CSV files |
| 2 | Audio preprocessing pipeline | Clean model-ready audio references |
| 3 | MFCC/mel baseline | Baseline metrics |
| 4 | wav2vec2/WavLM model | Trained deep model checkpoint |
| 5 | Evaluation harness | Accuracy, F1, ROC-AUC, EER, confusion matrix |
| 6 | Improvement experiments | Ablation results |
| 7 | Optional external/generalization test | Cross-dataset report |
| 8 | Streamlit demo app | Working local demo |
| 9 | Final report and presentation | Submission package |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Smoke Test

```bash
pytest
```

Dataset files and trained model checkpoints are intentionally not committed to GitHub.
