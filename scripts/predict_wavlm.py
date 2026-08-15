"""Run a saved WavLM detector on one local audio file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import WavLMForSequenceClassification

from src.data.audio import decode_audio_bytes, prepare_audio
from src.settings import load_config


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models/wavlm_detector"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    decoded = decode_audio_bytes(args.audio_file.read_bytes())
    waveform, attention_mask = prepare_audio(
        decoded,
        target_rate=int(config["audio"]["sample_rate"]),
        max_duration_seconds=float(config["audio"]["max_duration_seconds"]),
        crop_mode=str(config["audio"]["crop_mode"]),
        pad_value=float(config["audio"]["pad_value"]),
    )
    device = torch.device(args.device)
    model = WavLMForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
    with torch.inference_mode():
        logits = model(
            input_values=torch.from_numpy(waveform).unsqueeze(0).to(device),
            attention_mask=torch.from_numpy(attention_mask).unsqueeze(0).to(device),
        ).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    prediction = int(probabilities.argmax().item())
    label = model.config.id2label[prediction]
    print(f"Prediction: {label}")
    print(f"Bonafide score: {probabilities[0].item():.6f}")
    print(f"Spoof score: {probabilities[1].item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
