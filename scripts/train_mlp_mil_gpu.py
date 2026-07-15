from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from train_logistic_mil_gpu import (
    DEFAULT_DATA_DIR,
    ROOT,
    BagDataset,
    balanced_class_weights,
    class_counts,
    compute_loss,
    evaluate_binary_classifier,
    fit_standardizer,
    iter_folds,
    json_ready,
    load_and_prepare_data,
    make_loader,
    predict,
    resolve_device,
    select_threshold,
    set_seed,
    split_train_test,
    validate_dataset_for_experiment,
    validate_split_for_cross_validation,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mlp_mil_gpu"


class WindowMLP(nn.Module):
    """A nonlinear window classifier used inside the same MIL framework."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.network(windows).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CUDA PyTorch MLP MIL model on variable-length MFCC bags "
            "with a held-out group test set and five-fold group cross-validation."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_DATA_DIR / "mfcc_features.npy",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_DATA_DIR / "labels.npy",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="Optional source-recording group IDs aligned with the MFCC samples.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--hop-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-instance-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim-1", type=int, default=128)
    parser.add_argument("--hidden-dim-2", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive.")
    if args.hop_size <= 0 or args.hop_size > args.window_size:
        raise ValueError("--hop-size must be in [1, window_size].")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("--epochs, --batch-size and --patience must be positive.")
    if args.hidden_dim_1 <= 0 or args.hidden_dim_2 <= 0:
        raise ValueError("MLP hidden dimensions must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")


def make_dataset(
    indices: Sequence[int],
    mfcc_samples: Sequence[np.ndarray],
    deterministic_features: Sequence[np.ndarray],
    labels: np.ndarray,
    standardizer,
    args: argparse.Namespace,
    random_short_padding: bool,
    seed: int,
) -> BagDataset:
    return BagDataset(
        indices,
        mfcc_samples,
        deterministic_features,
        labels,
        standardizer,
        args.window_size,
        args.hop_size,
        random_short_padding=random_short_padding,
        seed=seed,
    )


def train_fold(
    fold_number: int,
    fold_train: np.ndarray,
    fold_validation: np.ndarray,
    test_indices: np.ndarray,
    mfcc_samples: Sequence[np.ndarray],
    deterministic_features: Sequence[np.ndarray],
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    standardizer = fit_standardizer(deterministic_features, fold_train)
    input_dim = int(standardizer.mean.shape[0])
    fold_seed = args.seed + fold_number

    train_dataset = make_dataset(
        fold_train,
        mfcc_samples,
        deterministic_features,
        labels,
        standardizer,
        args,
        random_short_padding=True,
        seed=fold_seed,
    )
    validation_dataset = make_dataset(
        fold_validation,
        mfcc_samples,
        deterministic_features,
        labels,
        standardizer,
        args,
        random_short_padding=False,
        seed=fold_seed,
    )
    test_dataset = make_dataset(
        test_indices,
        mfcc_samples,
        deterministic_features,
        labels,
        standardizer,
        args,
        random_short_padding=False,
        seed=fold_seed,
    )

    use_cuda = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=fold_seed,
        use_cuda=use_cuda,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=fold_seed,
        use_cuda=use_cuda,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=fold_seed,
        use_cuda=use_cuda,
    )

    model = WindowMLP(
        input_dim,
        hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    class_weights = balanced_class_weights(labels[fold_train]).to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for windows, mask, batch_labels, _ in train_loader:
            windows = windows.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(windows)
            loss, _ = compute_loss(
                logits,
                mask,
                batch_labels,
                args.top_k,
                class_weights,
                args.negative_instance_weight,
            )
            loss.backward()
            optimizer.step()

        model.eval()
        validation_loss_sum = 0.0
        validation_items = 0
        with torch.no_grad():
            for windows, mask, batch_labels, _ in validation_loader:
                windows = windows.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                logits = model(windows)
                validation_loss, _ = compute_loss(
                    logits,
                    mask,
                    batch_labels,
                    args.top_k,
                    class_weights,
                    args.negative_instance_weight,
                )
                validation_loss_sum += float(validation_loss.item()) * len(batch_labels)
                validation_items += len(batch_labels)

        mean_validation_loss = validation_loss_sum / validation_items
        print(
            f"fold={fold_number} epoch={epoch:03d} "
            f"validation_loss={mean_validation_loss:.6f}"
        )
        if mean_validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = mean_validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No MLP checkpoint was selected.")
    model.load_state_dict(best_state)

    validation_probabilities, _, validation_order = predict(
        model, validation_loader, device, args.top_k
    )
    test_probabilities, _, test_order = predict(model, test_loader, device, args.top_k)
    if not np.array_equal(test_order, test_indices):
        raise RuntimeError("Unexpected test prediction order.")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint = {
        "model_type": "window_mlp_mil",
        "model_state_dict": {
            key: value.detach().cpu() for key, value in best_state.items()
        },
        "standardizer_mean": standardizer.mean,
        "standardizer_scale": standardizer.scale,
        "input_dim": input_dim,
        "hidden_dims": [args.hidden_dim_1, args.hidden_dim_2],
        "dropout": args.dropout,
        "parameter_count": parameter_count,
        "window_size": args.window_size,
        "hop_size": args.hop_size,
        "top_k": args.top_k,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    return validation_probabilities, validation_order, test_probabilities, checkpoint


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    started_at = time.perf_counter()

    mfcc_samples, deterministic_features, labels, groups = load_and_prepare_data(
        args.features,
        args.labels,
        args.groups,
        args.window_size,
        args.hop_size,
        args.seed,
    )
    validate_dataset_for_experiment(labels, groups, args.folds)
    train_indices, test_indices = split_train_test(
        labels, groups, args.test_size, args.seed
    )
    validate_split_for_cross_validation(
        labels, train_indices, test_indices, args.folds, groups
    )

    if groups is None:
        print(
            "WARNING: no --groups file was provided; source-recording separation "
            "cannot be guaranteed."
        )
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"all_class_counts={class_counts(labels)}")
    print(f"train_class_counts={class_counts(labels[train_indices])}")
    print(f"test_class_counts={class_counts(labels[test_indices])}")

    out_of_fold_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    fold_test_probabilities: list[np.ndarray] = []
    checkpoints: list[dict[str, object]] = []

    for fold_number, (fold_train, fold_validation) in enumerate(
        iter_folds(train_indices, labels, groups, args.folds, args.seed), start=1
    ):
        (
            validation_probabilities,
            validation_order,
            test_probabilities,
            checkpoint,
        ) = train_fold(
            fold_number,
            fold_train,
            fold_validation,
            test_indices,
            mfcc_samples,
            deterministic_features,
            labels,
            args,
            device,
        )
        out_of_fold_probabilities[validation_order] = validation_probabilities
        fold_test_probabilities.append(test_probabilities)
        checkpoints.append(checkpoint)

    training_oof_probabilities = out_of_fold_probabilities[train_indices]
    if np.isnan(training_oof_probabilities).any():
        raise RuntimeError("Some training samples did not receive an OOF prediction.")

    threshold = select_threshold(labels[train_indices], training_oof_probabilities)
    oof_metrics = evaluate_binary_classifier(
        labels[train_indices], training_oof_probabilities, threshold
    )
    test_probabilities = np.mean(np.stack(fold_test_probabilities, axis=0), axis=0)
    test_metrics = evaluate_binary_classifier(
        labels[test_indices], test_probabilities, threshold
    )
    elapsed_seconds = time.perf_counter() - started_at

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold_number, checkpoint in enumerate(checkpoints, start=1):
        torch.save(checkpoint, args.output_dir / f"fold_{fold_number}.pt")
    np.save(args.output_dir / "train_indices.npy", train_indices)
    np.save(args.output_dir / "test_indices.npy", test_indices)
    np.save(args.output_dir / "oof_probabilities.npy", training_oof_probabilities)
    np.save(args.output_dir / "test_probabilities.npy", test_probabilities)

    results = {
        "model": {
            "type": "window_mlp_mil",
            "hidden_dims": [args.hidden_dim_1, args.hidden_dim_2],
            "dropout": args.dropout,
            "parameter_count": checkpoints[0]["parameter_count"],
        },
        "configuration": vars(args),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "threshold_source": "training out-of-fold predictions",
        "elapsed_seconds": elapsed_seconds,
        "fold_best_epochs": [checkpoint["best_epoch"] for checkpoint in checkpoints],
        "fold_best_validation_losses": [
            checkpoint["best_validation_loss"] for checkpoint in checkpoints
        ],
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_ready(results), file, ensure_ascii=False, indent=2)

    print("\nTraining OOF metrics:")
    print(json.dumps(json_ready(oof_metrics), ensure_ascii=False, indent=2))
    print("\nHeld-out test metrics:")
    print(json.dumps(json_ready(test_metrics), ensure_ascii=False, indent=2))
    print(f"\nElapsed seconds: {elapsed_seconds:.2f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
