# Phase 4 WavLM GPU Run

## Current Status

The WavLM training, validation, checkpoint, and inference pipeline is implemented. A real end-to-end smoke run passed locally with the pinned `microsoft/wavlm-base-plus` checkpoint and ASVspoof audio.

Local verification used four stratified training records, four development records, and one optimizer update. It proved that the real pretrained model can complete forward propagation, class-weighted backpropagation, validation, `safetensors` checkpoint saving, checkpoint reload, and standalone FLAC inference. Metrics from four records are deliberately not treated as research results.

The checkpoint has 94,579,314 parameters. Freezing the convolutional feature encoder leaves 90,378,866 parameters trainable. The saved model is approximately 378 MB.

The full 25,380-record training run has not been claimed as complete because this computer has CPU-only PyTorch. Phase 4 remains in progress until a GPU run produces the full checkpoint and development metrics.

## Technical Sources

- WavLM paper: https://arxiv.org/abs/2110.13900
- Official Microsoft checkpoint: https://huggingface.co/microsoft/wavlm-base-plus
- Transformers WavLM sequence-classification documentation: https://huggingface.co/docs/transformers/model_doc/wavlm

## Recommended GPU

- NVIDIA T4, L4, A10, A100, or similar CUDA GPU
- At least 15 GB GPU memory for the default batch size of 4
- Approximately 8 GB free storage for the repository, train/dev audio, caches, and checkpoint

If CUDA runs out of memory, use `--batch-size 2 --gradient-accumulation 8`. This keeps the same effective batch size of 16.

## Colab Commands

Select **Runtime > Change runtime type > T4 GPU**, then run:

```bash
git clone https://github.com/KJSK-Koushik/Audio-Deepfake-Detection-using-Self-Supervised-Speech-Representations.git
cd Audio-Deepfake-Detection-using-Self-Supervised-Speech-Representations
pip install "transformers>=4.57,<4.58" "safetensors>=0.4,<1" "pyarrow>=17,<25" "soundfile>=0.12,<1" "scipy>=1.13,<2" "scikit-learn>=1.5,<2" "pyyaml>=6,<7" "tqdm>=4.66,<5"
python -m scripts.download_asvspoof2019_hf --splits train dev
python -m scripts.train_wavlm --device cuda --epochs 3 --batch-size 4 --gradient-accumulation 4 --num-workers 2
```

The best checkpoint is written to `models/wavlm_detector/`. The directory contains:

- `model.safetensors`
- `config.json`
- `training_history.json`
- `training_summary.json`

Download or copy that complete directory back into this repository's local `models/` directory. Model files are intentionally ignored by Git.

An executable notebook with Google Drive checkpoint storage is provided at `notebooks/phase_4_wavlm_colab.ipynb`.

## Training Protocol

- WavLM checkpoint revision is pinned in `config.yaml`.
- The convolutional feature encoder is frozen.
- The Transformer and classification layers are fine-tuned.
- Class-weighted cross-entropy handles the real/fake imbalance.
- Mixed precision and gradient checkpointing reduce GPU memory use.
- AdamW uses a linear warmup and decay schedule.
- The best checkpoint is selected by development ROC-AUC.
- Evaluation data is not downloaded or accessed.

## Inference Check

After copying the trained model locally:

```powershell
$ProjectPython = "$env:LOCALAPPDATA\audio-deepfake-detection-venv\Scripts\python.exe"
& $ProjectPython -m scripts.predict_wavlm path\to\speech.flac --model-dir models\wavlm_detector
```

The command prints the predicted label and separate bonafide/spoof scores.
