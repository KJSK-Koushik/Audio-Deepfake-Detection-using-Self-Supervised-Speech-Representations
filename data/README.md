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

The command creates `metadata/train.csv`, `metadata/dev.csv`, `metadata/eval.csv`, and `metadata/summary.json`. The `processed/` directory is reserved for Phase 2.

The dataset license and redistribution rules apply. Audio and generated data are ignored by Git and must not be committed.

## Faster Audio Mirror

When Edinburgh DataShare is slow, the public `Bisher/ASVspoof_2019_LA` Hugging Face mirror provides the same three split sizes with embedded audio in Parquet files. Download one split or all splits with:

```powershell
.\scripts\download_asvspoof2019_hf.ps1 -Split train
.\scripts\download_asvspoof2019_hf.ps1 -Split all
```

The script pins the mirror revision and verifies each completed shard by its published SHA-256 hash. Phase 2 will convert/read these embedded audio records through a tested loader; the official ASVspoof protocols remain the source of labels and split definitions.
