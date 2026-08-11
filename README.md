# Audio Deepfake Detection using Self-Supervised Speech Representations

This project detects whether a speech audio sample is real human speech (`bonafide`) or fake/spoofed/generated speech (`spoof`).

[![CI](https://github.com/KJSK-Koushik/Audio-Deepfake-Detection-using-Self-Supervised-Speech-Representations/actions/workflows/ci.yml/badge.svg)](https://github.com/KJSK-Koushik/Audio-Deepfake-Detection-using-Self-Supervised-Speech-Representations/actions/workflows/ci.yml)

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

Implementation moves one phase at a time. A phase is reviewed and approved before work starts on the next one. See [docs/PHASES.md](docs/PHASES.md) for completion criteria.

Current gate: Phase 0 is complete. Phase 1 has not started.

## Repository Structure

```text
.
|-- config.yaml             # Shared experiment settings
|-- data/                   # Dataset instructions; audio is not committed
|-- docs/                   # Project specification and phase gates
|-- models/                 # Local model checkpoints (ignored by Git)
|-- notebooks/              # Optional exploration notebooks
|-- reports/                # Generated figures and result tables
|-- scripts/                # Setup and environment utilities
|-- src/
|   |-- data/               # Dataset indexing and preprocessing
|   |-- evaluation/         # Metrics and evaluation
|   |-- features/           # MFCC/mel/SSL feature extraction
|   `-- models/             # Baseline and deep models
`-- tests/                  # Automated tests
```

## Setup

```powershell
.\scripts\setup_environment.ps1
```

The repository path is long enough to break installation of some PyTorch files on Windows. The setup script therefore creates the isolated environment at `%LOCALAPPDATA%\audio-deepfake-detection-venv`.

Store its Python path for the current PowerShell session:

```powershell
$ProjectPython = "$env:LOCALAPPDATA\audio-deepfake-detection-venv\Scripts\python.exe"
```

Check the machine and installed ML packages:

```powershell
& $ProjectPython scripts\check_environment.py --strict
```

## Smoke Test

```powershell
& $ProjectPython -m ruff check .
& $ProjectPython -m pytest
```

Dataset files and trained model checkpoints are intentionally not committed to GitHub.

## CI/CD

GitHub Actions runs linting and smoke tests on every push and pull request to `main`. Model training is excluded because hosted CI is not the right place for long GPU experiments. Deployment will be added in Phase 8 after the Streamlit application exists; there is nothing useful to deploy during Phase 0.
