# Phase 3 MFCC Baseline Report

## Objective

Phase 3 establishes a simple, reproducible classical baseline. Its purpose is to show how well conventional speech features work before introducing the much larger self-supervised WavLM model.

## Experiment Design

| Component | Choice |
|---|---|
| Training data | Official ASVspoof 2019 LA train split (25,380) |
| Validation data | Official development split (24,844) |
| Input | Original valid speech region; no zero-padding included in features |
| Features | 40 MFCCs, delta, and delta-delta |
| Summary | Mean and standard deviation for each feature stream |
| Feature dimension | 240 values per recording |
| Classifier | StandardScaler + logistic regression |
| Imbalance handling | Class-balanced training weights |
| Random seed | 42 |
| Decision threshold | 0.5 |

No development recording is used for fitting, no random resplitting is performed, and no synthetic labels or augmented recordings are introduced.

## Results

| Metric | Majority-prior baseline | MFCC classifier |
|---|---:|---:|
| Accuracy | 89.74% | **91.44%** |
| Balanced accuracy | 50.00% | **84.63%** |
| Spoof precision | 89.74% | **97.15%** |
| Spoof recall | 100.00% | 93.20% |
| Spoof F1 | 94.59% | **95.13%** |
| Macro-F1 | 47.30% | **79.86%** |
| ROC-AUC | 50.00% | **94.27%** |
| EER (lower is better) | 50.00% | **13.18%** |

The majority baseline predicts every recording as spoof. Its 89.74% accuracy looks high only because the development split contains many more spoof recordings than bonafide recordings. Balanced accuracy, macro-F1, ROC-AUC, and EER show that it has no useful discrimination ability.

The MFCC model's development confusion matrix at threshold 0.5 is:

| Actual / predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 1,938 | 610 |
| Spoof | 1,516 | 20,780 |

This means the baseline correctly identifies 76.06% of bonafide recordings and 93.20% of spoof recordings at the default threshold. The classifier converged in 96 iterations.

## Runtime and Outputs

On the local Intel Core i7-1360P CPU with six extraction workers:

- Train feature extraction: 109.5 seconds
- Development feature extraction: 107.1 seconds
- Classifier training: 2.3 seconds
- Cached feature size: approximately 46 MB
- Saved model size: approximately 8 KB

Generated local outputs are ignored by Git:

- `data/processed/mfcc/train.npz`
- `data/processed/mfcc/dev.npz`
- `models/mfcc_logistic_regression.joblib`
- `reports/results/phase_3_mfcc_baseline.json`
- `reports/figures/phase_3_confusion_matrix.png`

Feature caches include a configuration and source-file fingerprint. They are rebuilt automatically if the dataset or MFCC settings change.

## Interpretation and Limitations

This is a useful baseline, but it is not the final system. Statistical MFCC summaries discard the detailed time sequence and may depend on artifacts specific to the six training attacks. The evaluation split contains different attack IDs and remains untouched until the formal comparison phase.

Phase 4 will test whether pretrained WavLM speech representations learn stronger and more transferable evidence than this MFCC baseline. Phase 4 has not started and requires explicit approval.
