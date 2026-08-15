"""Fine-tune WavLM for bonafide-versus-spoof speech classification."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from transformers import WavLMForSequenceClassification

from src.evaluation.binary import binary_classification_metrics
from src.models.wavlm import (
    balanced_class_weights,
    collate_audio_records,
    load_parquet_labels,
    make_audio_dataset,
    seed_everything,
)
from src.settings import ROOT_DIR, load_config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def selected_labels(dataset: object, parquet_path: Path) -> np.ndarray:
    labels = load_parquet_labels(parquet_path)
    if isinstance(dataset, Subset):
        return labels[np.asarray(dataset.indices)]
    return labels


def linear_warmup_decay(total_steps: int, warmup_ratio: float):
    warmup_steps = round(total_steps * warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return schedule


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
) -> tuple[float, dict[str, object]]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        input_values = batch["input_values"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        logits = model(input_values=input_values, attention_mask=attention_mask).logits
        losses.append(float(loss_function(logits, targets).item()))
        labels.append(targets.cpu().numpy())
        scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    metrics = binary_classification_metrics(np.concatenate(labels), np.concatenate(scores))
    return float(np.mean(losses)), metrics


def save_best_model(
    model: WavLMForSequenceClassification,
    output_dir: Path,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    config = load_config()
    wavlm_config = config["wavlm_training"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=int(wavlm_config["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(wavlm_config["batch_size"]))
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=int(wavlm_config["gradient_accumulation_steps"]),
    )
    parser.add_argument("--num-workers", type=int, default=int(wavlm_config["num_workers"]))
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "models" / "wavlm_detector")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation) <= 0:
        parser.error("epochs, batch-size, and gradient-accumulation must be positive")

    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    parquet_dir = ROOT_DIR / config["data"]["parquet_dir"]
    train_path = parquet_dir / "train.parquet"
    dev_path = parquet_dir / "dev.parquet"
    train_dataset = make_audio_dataset(
        train_path, config["audio"], limit=args.limit_train, seed=seed
    )
    dev_dataset = make_audio_dataset(dev_path, config["audio"], limit=args.limit_dev, seed=seed)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_audio_records,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_options)

    model_name = str(config["model"]["ssl_model_name"])
    revision = str(config["model"]["revision"])
    model = WavLMForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        num_labels=2,
        id2label={0: "bonafide", 1: "spoof"},
        label2id={"bonafide": 0, "spoof": 1},
        classifier_proj_size=int(config["model"]["classifier_proj_size"]),
    )
    if bool(wavlm_config["freeze_feature_encoder"]):
        model.freeze_feature_encoder()
    if bool(wavlm_config["gradient_checkpointing"]) and not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    train_labels = selected_labels(train_dataset, train_path)
    class_weights = balanced_class_weights(train_labels).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(wavlm_config["learning_rate"]),
        weight_decay=float(wavlm_config["weight_decay"]),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    planned_steps = updates_per_epoch * args.epochs
    total_steps = min(planned_steps, args.max_train_steps or planned_steps)
    scheduler = LambdaLR(
        optimizer,
        linear_warmup_decay(total_steps, float(wavlm_config["warmup_ratio"])),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_roc_auc = -math.inf
    best_epoch = 0
    best_summary: dict[str, object] | None = None
    global_step = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_values=input_values, attention_mask=attention_mask).logits
                loss = loss_function(logits, targets)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            epoch_losses.append(float(loss.item()))

            is_update = batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader)
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(wavlm_config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % int(wavlm_config["log_every_steps"]) == 0:
                    print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={np.mean(epoch_losses[-10:]):.4f}"
                    )
                if global_step >= total_steps:
                    break

        dev_loss, metrics = evaluate(model, dev_loader, device, loss_function)
        epoch_summary = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(epoch_losses)),
            "dev_loss": dev_loss,
            "metrics": metrics,
        }
        history.append(epoch_summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(epoch_summary, indent=2))
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = float(metrics["roc_auc"])
            best_epoch = epoch
            best_summary = {
                "status": "smoke" if args.max_train_steps else "trained",
                "model_name": model_name,
                "model_revision": revision,
                "device": str(device),
                "train_rows": len(train_dataset),
                "dev_rows": len(dev_dataset),
                "class_weights": class_weights.detach().cpu().tolist(),
                "best_epoch": epoch,
                "training_configuration": wavlm_config,
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
            }
            save_best_model(model, args.output_dir, best_summary)
        if global_step >= total_steps:
            break

    if best_summary is None:
        raise RuntimeError("Training ended without producing a checkpoint")
    best_summary["best_epoch"] = best_epoch
    best_summary["history"] = history
    best_summary["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(best_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best model written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
