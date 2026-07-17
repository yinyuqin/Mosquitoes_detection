from __future__ import annotations

import argparse
import csv
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from MIL_baseline import GlobalLogisticRegression
from MLP_baseline import GlobalMLP, make_global_features, trim_existing_zero_padding
from process_humbugdb_from_zip import extract_mfcc
from train_logistic_mil_gpu import (
    WindowLogisticRegression,
    aggregate_top_k_probabilities,
    make_window_features,
)
from train_mlp_mil_gpu import WindowMLP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOD_ROOT = ROOT / "data" / "raw"
DEFAULT_MODELS_ROOT = ROOT / "outputs"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "ood_evaluation"

ARCHIVES = (
    ("ood_negative/data1.zip", "negative_data1", 0),
    ("ood_negative/data2.zip", "negative_data2", 0),
    ("ood_positive/vasconcelos.zip", "positive_vasconcelos", 1),
    ("ood_positive/other.zip", "positive_other", 1),
)


@dataclass(frozen=True)
class OODSample:
    sample_id: str
    dataset: str
    archive: str
    member: str
    label: int
    source_sample_rate: int
    duration_seconds: float
    waveform_was_padded: bool
    mfcc: np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory: Path
    checkpoint_paths: tuple[Path, ...]
    threshold: float
    family: str
    window_size: int | None
    hop_size: int | None
    top_k: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every readable five-fold PyTorch model ensemble on the "
            "read-only positive and negative OOD audio archives."
        )
    )
    parser.add_argument("--ood-root", type=Path, default=DEFAULT_OOD_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--feature-window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--balance-positive-to-negative",
        action="store_true",
        help=(
            "After loading and validating every OOD file, randomly downsample valid "
            "positive samples without replacement to match the negative sample count."
        ),
    )
    parser.add_argument("--balance-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested, but CUDA is not available.")
    return device


def checkpoint_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return 10**9


def infer_model_family(checkpoint: dict[str, object]) -> str:
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict mapping.")

    keys = set(state_dict)
    is_window_model = all(key in checkpoint for key in ("window_size", "hop_size", "top_k"))
    if {"linear.weight", "linear.bias"}.issubset(keys):
        return "logistic_mil" if is_window_model else "logistic_global"
    if "network.0.weight" in keys and "network.8.weight" in keys:
        return "mlp_mil" if is_window_model else "mlp_global"
    raise ValueError(f"Unsupported checkpoint parameter keys: {sorted(keys)}")


def discover_models(models_root: Path) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for directory in sorted(path for path in models_root.iterdir() if path.is_dir()):
        checkpoint_paths = tuple(sorted(directory.glob("fold_*.pt"), key=checkpoint_number))
        metrics_path = directory / "metrics.json"
        if not checkpoint_paths:
            continue
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Weights found without metrics file: {directory}")

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        threshold = float(metrics["test_metrics"]["threshold"])
        checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
        family = infer_model_family(checkpoint)

        models.append(
            ModelSpec(
                name=directory.name,
                directory=directory,
                checkpoint_paths=checkpoint_paths,
                threshold=threshold,
                family=family,
                window_size=(int(checkpoint["window_size"]) if "window_size" in checkpoint else None),
                hop_size=(int(checkpoint["hop_size"]) if "hop_size" in checkpoint else None),
                top_k=(int(checkpoint["top_k"]) if "top_k" in checkpoint else None),
            )
        )
    if not models:
        raise FileNotFoundError(f"No readable fold_*.pt model ensembles found under {models_root}")
    return models


def load_ood_samples(args: argparse.Namespace) -> tuple[list[OODSample], list[dict[str, str]]]:
    samples: list[OODSample] = []
    failures: list[dict[str, str]] = []

    for relative_archive, dataset, label in ARCHIVES:
        archive_path = args.ood_root / relative_archive
        if not archive_path.is_file():
            raise FileNotFoundError(f"OOD archive does not exist: {archive_path}")

        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                member
                for member in archive.namelist()
                if not member.endswith("/")
                and Path(member).suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"}
            )
            for member in members:
                sample_id = f"{dataset}::{member}"
                try:
                    audio_bytes = archive.read(member)
                    info = sf.info(io.BytesIO(audio_bytes))
                    source_sample_rate = int(info.samplerate)
                    duration_seconds = float(info.duration)
                    waveform, loaded_sample_rate = librosa.load(
                        io.BytesIO(audio_bytes), sr=args.sample_rate, mono=True
                    )
                    if loaded_sample_rate != args.sample_rate:
                        raise RuntimeError(
                            f"Expected {args.sample_rate} Hz after loading, got {loaded_sample_rate}."
                        )
                    if len(waveform) == 0:
                        raise ValueError("Audio file contains zero samples.")
                    waveform_was_padded = len(waveform) < args.n_fft
                    if waveform_was_padded:
                        missing = args.n_fft - len(waveform)
                        left = missing // 2
                        waveform = np.pad(waveform, (left, missing - left), mode="constant")
                    mfcc = extract_mfcc(
                        waveform,
                        loaded_sample_rate,
                        n_mfcc=args.n_mfcc,
                        hop_length=args.hop_length,
                        n_fft=args.n_fft,
                        window_size=args.feature_window_size,
                    )
                    if mfcc is None:
                        raise ValueError("Audio is too short for MFCC extraction.")
                    mfcc = trim_existing_zero_padding(np.asarray(mfcc, dtype=np.float32))
                    if not np.isfinite(mfcc).all():
                        raise ValueError("Extracted MFCC contains NaN or infinity.")
                    samples.append(
                        OODSample(
                            sample_id=sample_id,
                            dataset=dataset,
                            archive=relative_archive,
                            member=member,
                            label=label,
                            source_sample_rate=source_sample_rate,
                            duration_seconds=duration_seconds,
                            waveform_was_padded=waveform_was_padded,
                            mfcc=mfcc,
                        )
                    )
                except Exception as error:
                    failures.append(
                        {"sample_id": sample_id, "archive": relative_archive, "error": str(error)}
                    )
        print(f"loaded dataset={dataset} samples={sum(s.dataset == dataset for s in samples)}")
    return samples, failures


def balance_positive_samples(
    samples: Sequence[OODSample], seed: int
) -> tuple[list[OODSample], dict[str, object]]:
    negative_indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample.label == 0], dtype=np.int64
    )
    positive_indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample.label == 1], dtype=np.int64
    )
    if len(positive_indices) < len(negative_indices):
        raise ValueError(
            "Positive downsampling cannot balance this dataset because there are fewer "
            "valid positives than negatives."
        )

    rng = np.random.default_rng(seed)
    selected_positive_indices = rng.choice(
        positive_indices, size=len(negative_indices), replace=False
    )
    selected_indices = np.sort(np.concatenate([negative_indices, selected_positive_indices]))
    selected_samples = [samples[int(index)] for index in selected_indices]
    selected_dataset_counts = {
        dataset: int(sum(sample.dataset == dataset for sample in selected_samples))
        for dataset in sorted({sample.dataset for sample in samples})
    }
    selection = {
        "enabled": True,
        "method": "uniform random positive downsampling without replacement",
        "seed": seed,
        "positive_candidates": int(len(positive_indices)),
        "negative_samples_retained": int(len(negative_indices)),
        "positive_samples_selected": int(len(selected_positive_indices)),
        "selected_dataset_counts": selected_dataset_counts,
    }
    return selected_samples, selection


def build_model(checkpoint: dict[str, object], family: str) -> nn.Module:
    input_dim = int(checkpoint["input_dim"])
    if family == "logistic_global":
        return GlobalLogisticRegression(input_dim)
    if family == "logistic_mil":
        return WindowLogisticRegression(input_dim)

    hidden_dims = tuple(int(value) for value in checkpoint["hidden_dims"])
    if len(hidden_dims) != 2:
        raise ValueError(f"Expected two MLP hidden dimensions, got {hidden_dims}.")
    dropout = float(checkpoint["dropout"])
    if family == "mlp_global":
        return GlobalMLP(input_dim, hidden_dims=hidden_dims, dropout=dropout)
    if family == "mlp_mil":
        return WindowMLP(input_dim, hidden_dims=hidden_dims, dropout=dropout)
    raise ValueError(f"Unsupported model family: {family}")


def predict_global_fold(
    model: nn.Module,
    features: np.ndarray,
    checkpoint: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    mean = np.asarray(checkpoint["standardizer_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["standardizer_scale"], dtype=np.float32)
    standardized = ((features - mean) / scale).astype(np.float32, copy=False)
    predictions: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(standardized), batch_size):
            batch = torch.from_numpy(standardized[start : start + batch_size]).to(device)
            predictions.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def predict_mil_fold(
    model: nn.Module,
    features: Sequence[np.ndarray],
    checkpoint: dict[str, object],
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> np.ndarray:
    mean = np.asarray(checkpoint["standardizer_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["standardizer_scale"], dtype=np.float32)
    predictions: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            raw_batch = features[start : start + batch_size]
            batch = [((item - mean) / scale).astype(np.float32, copy=False) for item in raw_batch]
            lengths = [len(item) for item in batch]
            max_windows = max(lengths)
            feature_dim = int(batch[0].shape[1])
            padded = torch.zeros((len(batch), max_windows, feature_dim), dtype=torch.float32)
            mask = torch.zeros((len(batch), max_windows), dtype=torch.bool)
            for row, item in enumerate(batch):
                padded[row, : len(item)] = torch.from_numpy(item)
                mask[row, : len(item)] = True

            logits = model(padded.to(device))
            probabilities = aggregate_top_k_probabilities(logits, mask.to(device), top_k)
            predictions.append(probabilities.cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def predict_model(
    spec: ModelSpec,
    global_features: np.ndarray,
    window_features: dict[tuple[int, int], list[np.ndarray]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    fold_probabilities: list[np.ndarray] = []
    for checkpoint_path in spec.checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if infer_model_family(checkpoint) != spec.family:
            raise ValueError(f"Inconsistent model family in {checkpoint_path}")
        model = build_model(checkpoint, spec.family).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if spec.family.endswith("global"):
            probabilities = predict_global_fold(
                model, global_features, checkpoint, device, batch_size
            )
        else:
            if spec.window_size is None or spec.hop_size is None or spec.top_k is None:
                raise ValueError(f"MIL configuration is incomplete for {spec.name}")
            probabilities = predict_mil_fold(
                model,
                window_features[(spec.window_size, spec.hop_size)],
                checkpoint,
                device,
                batch_size,
                spec.top_k,
            )
        fold_probabilities.append(probabilities)
        del model
    return np.mean(np.stack(fold_probabilities, axis=0), axis=0)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result: dict[str, object] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "sample_count": int(len(labels)),
        "class_counts": class_counts(labels),
        "mean_probability": float(probabilities.mean()),
    }
    return result


def subset_metrics(label: int, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    positive_rate = float(np.mean(probabilities >= threshold))
    result: dict[str, object] = {
        "label": label,
        "sample_count": int(len(probabilities)),
        "threshold": threshold,
        "mean_probability": float(probabilities.mean()),
        "median_probability": float(np.median(probabilities)),
        "positive_prediction_rate": positive_rate,
    }
    if label == 0:
        result["specificity"] = 1.0 - positive_rate
        result["false_positive_rate"] = positive_rate
    else:
        result["recall_sensitivity"] = positive_rate
        result["false_negative_rate"] = 1.0 - positive_rate
    return result


def write_predictions(
    path: Path,
    samples: Sequence[OODSample],
    models: Sequence[ModelSpec],
    probabilities: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "sample_id",
        "dataset",
        "archive",
        "member",
        "label",
        "source_sample_rate",
        "duration_seconds",
        "waveform_was_padded",
    ]
    for model in models:
        fieldnames.extend([f"{model.name}_probability", f"{model.name}_prediction"])

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample in enumerate(samples):
            row: dict[str, object] = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "archive": sample.archive,
                "member": sample.member,
                "label": sample.label,
                "source_sample_rate": sample.source_sample_rate,
                "duration_seconds": sample.duration_seconds,
                "waveform_was_padded": sample.waveform_was_padded,
            }
            for model in models:
                probability = float(probabilities[model.name][index])
                row[f"{model.name}_probability"] = probability
                row[f"{model.name}_prediction"] = int(probability >= model.threshold)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    started_at = time.perf_counter()

    models = discover_models(args.models_root)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print("models=" + ",".join(model.name for model in models))

    loaded_samples, failures = load_ood_samples(args)
    if not loaded_samples:
        raise RuntimeError("No valid OOD audio samples were loaded.")
    loaded_class_counts = class_counts(
        np.asarray([sample.label for sample in loaded_samples], dtype=np.int64)
    )
    if args.balance_positive_to_negative:
        samples, selection = balance_positive_samples(loaded_samples, args.balance_seed)
    else:
        samples = loaded_samples
        selection = {
            "enabled": False,
            "method": "no class downsampling",
            "seed": None,
            "positive_candidates": int(loaded_class_counts.get("1", 0)),
            "negative_samples_retained": int(loaded_class_counts.get("0", 0)),
            "positive_samples_selected": int(loaded_class_counts.get("1", 0)),
        }
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    print(f"evaluation_class_counts={class_counts(labels)}")
    global_features = np.stack([make_global_features(sample.mfcc) for sample in samples])

    rng = np.random.default_rng(0)
    window_features: dict[tuple[int, int], list[np.ndarray]] = {}
    for model in models:
        if model.window_size is None or model.hop_size is None:
            continue
        key = (model.window_size, model.hop_size)
        if key not in window_features:
            window_features[key] = [
                make_window_features(
                    sample.mfcc,
                    window_size=model.window_size,
                    hop_size=model.hop_size,
                    random_short_padding=False,
                    rng=rng,
                )
                for sample in samples
            ]

    all_probabilities: dict[str, np.ndarray] = {}
    model_results: dict[str, object] = {}
    datasets = sorted({sample.dataset for sample in samples})
    for model in models:
        model_started_at = time.perf_counter()
        probabilities = predict_model(
            model, global_features, window_features, device, args.batch_size
        )
        all_probabilities[model.name] = probabilities
        by_dataset: dict[str, object] = {}
        for dataset in datasets:
            indices = np.asarray(
                [index for index, sample in enumerate(samples) if sample.dataset == dataset],
                dtype=np.int64,
            )
            by_dataset[dataset] = subset_metrics(
                int(labels[indices][0]), probabilities[indices], model.threshold
            )

        metrics = binary_metrics(labels, probabilities, model.threshold)
        model_results[model.name] = {
            "family": model.family,
            "weights_directory": str(model.directory),
            "fold_count": len(model.checkpoint_paths),
            "ensemble": "mean probability across folds",
            "threshold_source": "training out-of-fold predictions; OOD was not used for tuning",
            "overall": metrics,
            "by_dataset": by_dataset,
            "elapsed_seconds": time.perf_counter() - model_started_at,
        }
        print(
            f"model={model.name} f1={metrics['f1']:.6f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.6f} "
            f"recall={metrics['recall_sensitivity']:.6f} "
            f"fpr={metrics['false_positive_rate']:.6f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output_dir / "predictions.csv", samples, models, all_probabilities)
    np.savez_compressed(
        args.output_dir / "probabilities.npz",
        labels=labels,
        sample_ids=np.asarray([sample.sample_id for sample in samples]),
        **all_probabilities,
    )

    dataset_counts = {
        dataset: int(sum(sample.dataset == dataset for sample in samples)) for dataset in datasets
    }
    results = {
        "protocol": {
            "purpose": "read-only OOD evaluation",
            "threshold_policy": "reuse each model's training OOF threshold without OOD tuning",
            "fold_ensemble": "mean probability across all readable fold checkpoints",
            "sample_rate": args.sample_rate,
            "n_mfcc": args.n_mfcc,
            "hop_length": args.hop_length,
            "n_fft": args.n_fft,
            "mfcc_channels": args.n_mfcc * 3,
            "global_feature_dimension": int(global_features.shape[1]),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "dataset": {
            "sample_count": len(samples),
            "class_counts": class_counts(labels),
            "loaded_valid_sample_count_before_selection": len(loaded_samples),
            "loaded_valid_class_counts_before_selection": loaded_class_counts,
            "selection": selection,
            "dataset_counts": dataset_counts,
            "failed_sample_count": len(failures),
            "short_audio_padded_count": int(sum(sample.waveform_was_padded for sample in samples)),
            "failures": failures,
            "data2_independence_note": (
                "negative_data2 contains segments from one origin recording; sample-level "
                "metrics must not be interpreted as 100 independent recordings."
            ),
        },
        "models": model_results,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"elapsed_seconds={results['elapsed_seconds']:.2f}")
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
