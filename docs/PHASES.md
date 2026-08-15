# Implementation Phases and Approval Gates

Work stops at the end of every phase for review. The next phase starts only after explicit approval.

| Phase | What gets built | Output | Completion gate |
|---:|---|---|---|
| 0 | Repository, environment, configuration, structure, CI | Working skeleton | Local checks and GitHub CI pass |
| 1 | ASVspoof 2019 LA setup and protocol parser | Train/dev/eval metadata CSV files | Counts and labels match official protocols |
| 2 | Loading and preprocessing at mono 16 kHz | Model-ready audio loader and data report | Audio/label validation tests pass |
| 3 | MFCC or mel baseline | Baseline model and metrics | Reproducible evaluation report exists |
| 4 | WavLM/wav2vec2 detector | Deep model checkpoint and logs | Training and inference work end to end |
| 5 | Complete evaluation harness | Metrics, ROC, EER, confusion matrix | Baseline and deep model are compared fairly |
| 6 | Augmentation and tuning experiments | Improved checkpoint and ablation table | Each claimed improvement has evidence |
| 7 | Optional external evaluation | Cross-dataset report | Generalization limits are documented |
| 8 | Streamlit upload and prediction app | Working demo | Valid/invalid upload paths are tested |
| 9 | Report, slides, diagrams, viva notes | Submission package | Results match code and saved artifacts |

## Current Status

- Phase 0: complete (local environment, smoke tests, and CI skeleton verified)
- Phase 1: complete (official protocol metadata and integrity-checked download tooling verified)
- Phase 2: complete (all downloaded audio decoded and model-ready loading verified)
- Phase 3: complete (full MFCC baseline and reproducible development metrics verified)
- Phase 4: waiting for explicit approval
- Phases 5-9: not started
