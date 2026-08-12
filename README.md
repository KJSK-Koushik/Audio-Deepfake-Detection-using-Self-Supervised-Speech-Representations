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

Current gate: Phase 3 is complete. Phase 4 has not started and requires explicit approval.

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

## Phase 1 Dataset Setup

Create the official ASVspoof 2019 LA metadata indexes without requiring the audio download:

```powershell
& $ProjectPython -m scripts.fetch_asvspoof2019_protocols
& $ProjectPython -m scripts.build_asvspoof2019_metadata --skip-audio-check
```

Download checksum-verified audio containers from the pinned mirror as needed:

```powershell
.\scripts\download_asvspoof2019_hf.ps1 -Split train
.\scripts\download_asvspoof2019_hf.ps1 -Split dev
```

The evaluation split can be downloaded later with `-Split eval`. See
[docs/PHASE_1_REPORT.md](docs/PHASE_1_REPORT.md) for provenance, counts, and validation details.

## Phase 2 Audio Preprocessing

Audit every downloaded train and development recording and create local row manifests:

```powershell
& $ProjectPython -m scripts.audit_asvspoof2019_audio
```

The loader in `src/data/audio.py` decodes embedded FLAC audio, mixes stereo inputs to mono,
resamples when necessary, and returns a fixed 6-second waveform with an attention mask. Short
recordings are zero-padded and longer recordings are center-cropped. Processing happens on demand,
so the project does not store a second large copy of the audio.

See [docs/PHASE_2_REPORT.md](docs/PHASE_2_REPORT.md) for the complete audit results.

## Phase 3 MFCC Baseline

Extract cached MFCC statistics, train the class-balanced logistic regression baseline, and evaluate
it on the official development split:

```powershell
& $ProjectPython -m scripts.run_mfcc_baseline --jobs 6 --batch-size 500
```

The full run uses all 25,380 training and 24,844 development recordings. It reports both ordinary
classification metrics and imbalance-aware metrics such as balanced accuracy, macro-F1, ROC-AUC,
and EER. See [docs/PHASE_3_REPORT.md](docs/PHASE_3_REPORT.md) for results and interpretation.

## CI/CD

GitHub Actions runs linting and unit tests on every push and pull request to `main`. Tests include
synthetic FLAC decoding and preprocessing without downloading the dataset in CI. Model training is
excluded because hosted CI is not the right place for long GPU experiments. Deployment will be
added in Phase 8 after the Streamlit application exists.
