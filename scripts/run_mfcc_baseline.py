"""Extract MFCC features and train the Phase 3 classical baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.audio import ASVspoofParquetDataset
from src.evaluation.binary import binary_classification_metrics
from src.features.mfcc import MFCCConfig, extract_mfcc_statistics, feature_names
from src.settings import ROOT_DIR, load_config


def feature_cache_key(
    parquet_path: Path,
    audio_config: dict[str, object],
    mfcc_config: MFCCConfig,
) -> str:
    source = parquet_path.stat()
    payload = {
        "parquet_size": source.st_size,
        "parquet_modified_ns": source.st_mtime_ns,
        "audio_config": {
            key: audio_config[key]
            for key in ("sample_rate", "max_duration_seconds", "crop_mode", "pad_value")
        },
        "mfcc_config": mfcc_config.__dict__,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_batch(
    parquet_path: Path,
    indices: np.ndarray,
    audio_config: dict[str, object],
    mfcc_config: MFCCConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = ASVspoofParquetDataset(
        parquet_path,
        target_rate=int(audio_config["sample_rate"]),
        max_duration_seconds=float(audio_config["max_duration_seconds"]),
        crop_mode=str(audio_config["crop_mode"]),
        pad_value=float(audio_config["pad_value"]),
    )
    rows = len(indices)
    features = np.empty((rows, 6 * mfcc_config.n_mfcc), dtype=np.float32)
    labels = np.empty(rows, dtype=np.int8)
    utterance_ids = np.empty(rows, dtype="U12")

    for output_index, dataset_index in enumerate(indices):
        record = dataset[dataset_index]
        valid_waveform = record["waveform"][record["attention_mask"].astype(bool)]
        features[output_index] = extract_mfcc_statistics(valid_waveform, mfcc_config)
        labels[output_index] = record["label_id"]
        utterance_ids[output_index] = record["utterance_id"]
    return features, labels, utterance_ids


def extract_split_features(
    split: str,
    parquet_path: Path,
    cache_path: Path,
    audio_config: dict[str, object],
    mfcc_config: MFCCConfig,
    *,
    jobs: int,
    batch_size: int,
    limit: int | None = None,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cache_key = feature_cache_key(parquet_path, audio_config, mfcc_config)
    if cache_path.is_file() and not force and limit is None:
        cached = np.load(cache_path)
        cached_key = str(cached["cache_key"].item()) if "cache_key" in cached.files else None
        if cached_key == cache_key:
            return cached["features"], cached["labels"], cached["utterance_ids"], 0.0
        print(f"{split}: feature configuration changed; rebuilding cache")

    dataset_size = len(ASVspoofParquetDataset(parquet_path))
    rows = dataset_size if limit is None else min(limit, dataset_size)
    indices = (
        np.arange(dataset_size)
        if limit is None
        else np.linspace(0, dataset_size - 1, rows, dtype=np.int64)
    )
    batches_of_indices = [indices[start : start + batch_size] for start in range(0, rows, batch_size)]

    started = time.perf_counter()
    batches = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
        delayed(extract_batch)(parquet_path, batch, audio_config, mfcc_config)
        for batch in batches_of_indices
    )
    elapsed = time.perf_counter() - started
    features = np.concatenate([batch[0] for batch in batches])
    labels = np.concatenate([batch[1] for batch in batches])
    utterance_ids = np.concatenate([batch[2] for batch in batches])

    if limit is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=features,
            labels=labels,
            utterance_ids=utterance_ids,
            cache_key=np.asarray(cache_key),
        )
    print(f"{split}: extracted {rows:,} rows in {elapsed:.1f} seconds")
    return features, labels, utterance_ids, elapsed


def plot_confusion_matrix(matrix: list[list[int]], destination: Path) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{values[row, column]:,}", ha="center", va="center")
    axis.set(
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Bonafide", "Spoof"),
        yticklabels=("Bonafide", "Spoof"),
        xlabel="Predicted label",
        ylabel="True label",
        title="MFCC baseline: development confusion matrix",
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Use the first N rows per split for a smoke run")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    if args.jobs == 0 or args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("jobs must be non-zero; batch-size and limit must be positive")

    seed = int(config["training"]["seed"])
    baseline_config = config["baseline"]
    mfcc_config = MFCCConfig(
        sample_rate=int(config["audio"]["sample_rate"]),
        n_mfcc=int(baseline_config["n_mfcc"]),
        n_mels=int(baseline_config["n_mels"]),
        n_fft=int(baseline_config["n_fft"]),
        hop_length=int(baseline_config["hop_length"]),
        fmin=float(baseline_config["fmin"]),
        fmax=float(baseline_config["fmax"]),
    )
    parquet_dir = ROOT_DIR / config["data"]["parquet_dir"]
    feature_dir = ROOT_DIR / config["data"]["processed_dir"] / "mfcc"

    split_data = {}
    extraction_seconds = {}
    for split in ("train", "dev"):
        features, labels, utterance_ids, elapsed = extract_split_features(
            split,
            parquet_dir / f"{split}.parquet",
            feature_dir / f"{split}.npz",
            config["audio"],
            mfcc_config,
            jobs=args.jobs,
            batch_size=args.batch_size,
            limit=args.limit,
            force=args.force_features,
        )
        split_data[split] = (features, labels, utterance_ids)
        extraction_seconds[split] = elapsed

    train_features, train_labels, _ = split_data["train"]
    dev_features, dev_labels, _ = split_data["dev"]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=int(baseline_config["max_iter"]),
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    training_started = time.perf_counter()
    model.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - training_started
    scores = model.predict_proba(dev_features)[:, 1]
    metrics = binary_classification_metrics(dev_labels, scores)

    dummy = DummyClassifier(strategy="prior", random_state=seed)
    dummy.fit(train_features, train_labels)
    dummy_scores = dummy.predict_proba(dev_features)[:, 1]
    dummy_metrics = binary_classification_metrics(dev_labels, dummy_scores)

    result = {
        "experiment": "phase_3_mfcc_logistic_regression",
        "seed": seed,
        "feature_dimension": len(feature_names(mfcc_config.n_mfcc)),
        "mfcc_config": mfcc_config.__dict__,
        "train_rows": len(train_labels),
        "dev_rows": len(dev_labels),
        "train_label_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(train_labels, return_counts=True), strict=True)
        },
        "dev_label_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(dev_labels, return_counts=True), strict=True)
        },
        "label_mapping": config["data"]["labels"],
        "extraction_seconds": extraction_seconds,
        "training_seconds": training_seconds,
        "model": {
            "type": "StandardScaler + class-balanced LogisticRegression",
            "iterations": model.named_steps["classifier"].n_iter_.tolist(),
            "metrics": metrics,
        },
        "majority_prior_baseline": dummy_metrics,
    }

    if args.limit is None:
        model_path = ROOT_DIR / "models" / "mfcc_logistic_regression.joblib"
        result_path = ROOT_DIR / "reports" / "results" / "phase_3_mfcc_baseline.json"
        figure_path = ROOT_DIR / "reports" / "figures" / "phase_3_confusion_matrix.png"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        plot_confusion_matrix(metrics["confusion_matrix"], figure_path)
        print(f"Model written to: {model_path}")
        print(f"Results written to: {result_path}")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
