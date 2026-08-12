# Data Directory

Phase 1 uses the official ASVspoof 2019 Logical Access archive from the University of Edinburgh DataShare:

- Landing page: https://datashare.ed.ac.uk/handle/10283/3336
- Archive: `LA.zip` (approximately 7.64 GB)
- License: Open Data Commons Attribution License (ODC-By 1.0)

Run the resumable downloader from the repository root:

```powershell
.\scripts\download_asvspoof2019_la.ps1
```

After extraction, the relevant local layout is:

```text
data/raw/asvspoof2019_la/
`-- LA/
    |-- ASVspoof2019_LA_cm_protocols/
    |-- ASVspoof2019_LA_train/flac/
    |-- ASVspoof2019_LA_dev/flac/
    `-- ASVspoof2019_LA_eval/flac/
```

If the full archive download is still in progress, fetch the official protocol content from the checksum-pinned National Institute of Informatics mirror:

```powershell
$ProjectPython = "$env:LOCALAPPDATA\audio-deepfake-detection-venv\Scripts\python.exe"
& $ProjectPython -m scripts.fetch_asvspoof2019_protocols
```

Generate and validate metadata:

```powershell
& $ProjectPython -m scripts.build_asvspoof2019_metadata
```

The command creates `metadata/train.csv`, `metadata/dev.csv`, `metadata/eval.csv`, and `metadata/summary.json`.

The dataset license and redistribution rules apply. Audio and generated data are ignored by Git and must not be committed.

## Faster Audio Mirror

When Edinburgh DataShare is slow, the public `Bisher/ASVspoof_2019_LA` Hugging Face mirror provides the same three split sizes with embedded audio in Parquet files. Download one split or all splits with:

```powershell
.\scripts\download_asvspoof2019_hf.ps1 -Split train
.\scripts\download_asvspoof2019_hf.ps1 -Split all
```

The script pins the mirror revision and verifies each completed shard by its published SHA-256 hash. The official ASVspoof protocols remain the source of labels and split definitions.

## Audio Audit and Model-Ready References

Run the complete Phase 2 audit after downloading train and development audio:

```powershell
& $ProjectPython -m scripts.audit_asvspoof2019_audio
```

Generated files under `data/processed/` are intentionally ignored by Git:

- `audio_audit.json`: decoding, sample-rate, channel, label, duration, and amplitude summary
- `train_audio_manifest.csv`: train Parquet row references and source audio properties
- `dev_audio_manifest.csv`: development Parquet row references and source audio properties

The manifests reference original Parquet rows instead of duplicating padded audio. The loader applies
mono conversion, resampling, deterministic center cropping, and zero padding when a sample is read.
