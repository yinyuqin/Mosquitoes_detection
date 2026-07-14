from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import get_worker_info


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "humbugdb_mfcc"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "logistic_mil_gpu"


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale).astype(np.float32, copy=False)


class WindowLogisticRegression(nn.Module):
    """A single linear logit per MFCC window (GPU logistic regression)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.linear(windows).squeeze(-1)


class BagDataset(Dataset):
    """One item is one recording/segment containing a variable number of windows."""

    def __init__(
        self,
        indices: Sequence[int],
        mfcc_samples: Sequence[np.ndarray],
        deterministic_features: Sequence[np.ndarray],
        labels: np.ndarray,
        standardizer: Standardizer,
        window_size: int,
        hop_size: int,
        random_short_padding: bool,
        seed: int,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mfcc_samples = mfcc_samples
        self.deterministic_features = deterministic_features
        self.labels = labels
        self.standardizer = standardizer
        self.window_size = window_size
        self.hop_size = hop_size
        self.random_short_padding = random_short_padding
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_index = int(self.indices[position])
        mfcc = self.mfcc_samples[sample_index]

        # Long samples are precomputed once. Short samples are regenerated during
        # training so their zero padding is randomly split between left and right.
        if self.random_short_padding and mfcc.shape[1] < self.window_size:
            features = make_window_features(
                mfcc,
                window_size=self.window_size,
                hop_size=self.hop_size,
                random_short_padding=True,
                rng=self.rng,
            )
        else:
            features = self.deterministic_features[sample_index]

        features = self.standardizer.transform(features)
        return (
            torch.from_numpy(features),
            torch.tensor(float(self.labels[sample_index]), dtype=torch.float32),
            sample_index,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CUDA PyTorch logistic MIL model on variable-length MFCC bags "
            "with a held-out test set and five-fold cross-validation."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_DATA_DIR / "mfcc_features.npy",
        help="Object-array .npy file containing MFCC matrices [channels, frames].",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_DATA_DIR / "labels.npy",
        help="Binary recording/segment labels aligned with --features.",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help=(
            "Optional .npy group IDs (for example original recording IDs). When "
            "provided, no group may cross train/validation/test boundaries."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument(
        "--hop-size",
        type=int,
        default=32,
        help="Window hop in MFCC frames; 32 with a 64-frame window gives 50%% overlap.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-instance-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device. Defaults to CUDA and deliberately does not silently fall back.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and check the NVIDIA driver."
        )
    return device


def trim_existing_zero_padding(mfcc: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Remove only trailing all-zero frames added by the existing preprocessor."""
    nonzero_columns = np.any(np.abs(mfcc) > tolerance, axis=0)
    nonzero_positions = np.flatnonzero(nonzero_columns)
    if len(nonzero_positions) == 0:
        return mfcc[:, :1]
    return mfcc[:, : int(nonzero_positions[-1]) + 1]


def make_window_features(
    mfcc: np.ndarray,
    window_size: int,
    hop_size: int,
    random_short_padding: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Split long MFCCs with overlap and summarize each fixed-size window.

    A short sample is padded to exactly one window. During training, the total
    zero padding is randomly distributed between the left and right sides. For
    validation/test, padding is centered and therefore deterministic.
    """
    channels, frames = mfcc.shape
    if frames < window_size:
        missing = window_size - frames
        if random_short_padding:
            left = int(rng.integers(0, missing + 1))
        else:
            left = missing // 2
        right = missing - left
        padded = np.pad(mfcc, ((0, 0), (left, right)), mode="constant")
        windows = padded[None, :, :]
    else:
        last_start = frames - window_size
        starts = list(range(0, last_start + 1, hop_size))
        if starts[-1] != last_start:
            starts.append(last_start)
        windows = np.stack(
            [mfcc[:, start : start + window_size] for start in starts], axis=0
        )

    means = windows.mean(axis=2)
    standard_deviations = windows.std(axis=2)
    features = np.concatenate([means, standard_deviations], axis=1)
    if features.shape[1] != channels * 2:
        raise RuntimeError("Unexpected window feature dimension.")
    return features.astype(np.float32, copy=False)


def load_and_prepare_data(
    features_path: Path,
    labels_path: Path,
    groups_path: Path | None,
    window_size: int,
    hop_size: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray | None]:
    raw_features = np.load(features_path, allow_pickle=True)
    labels = np.asarray(np.load(labels_path), dtype=np.int64)
    groups = None if groups_path is None else np.asarray(np.load(groups_path))

    if len(raw_features) != len(labels):
        raise ValueError("Feature and label counts do not match.")
    if groups is not None and len(groups) != len(labels):
        raise ValueError("Group and label counts do not match.")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Both binary classes 0 and 1 must be present.")

    rng = np.random.default_rng(seed)
    prepared_mfcc: list[np.ndarray] = []
    deterministic_features: list[np.ndarray] = []
    expected_channels: int | None = None

    for sample_index, sample in enumerate(raw_features):
        mfcc = np.asarray(sample, dtype=np.float32)
        if mfcc.ndim != 2 or mfcc.shape[0] == 0 or mfcc.shape[1] == 0:
            raise ValueError(
                f"MFCC sample {sample_index} must have shape [channels, frames]."
            )
        if not np.isfinite(mfcc).all():
            raise ValueError(f"MFCC sample {sample_index} contains NaN or infinity.")
        if expected_channels is None:
            expected_channels = int(mfcc.shape[0])
        elif mfcc.shape[0] != expected_channels:
            raise ValueError("Every MFCC sample must use the same number of channels.")

        mfcc = trim_existing_zero_padding(mfcc)
        prepared_mfcc.append(mfcc)
        deterministic_features.append(
            make_window_features(
                mfcc,
                window_size=window_size,
                hop_size=hop_size,
                random_short_padding=False,
                rng=rng,
            )
        )

    return prepared_mfcc, deterministic_features, labels, groups


def class_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def validate_dataset_for_experiment(
    labels: np.ndarray, groups: np.ndarray | None, folds: int
) -> None:
    counts = class_counts(labels)
    minimum_per_class = folds + 1  # at least `folds` in train and one in test
    if min(counts.get(0, 0), counts.get(1, 0)) < minimum_per_class:
        raise ValueError(
            f"A held-out test set plus {folds}-fold CV requires at least "
            f"{minimum_per_class} samples of each class; got {counts}."
        )
    if groups is not None:
        for label in (0, 1):
            label_group_count = len(np.unique(groups[labels == label]))
            if label_group_count < minimum_per_class:
                raise ValueError(
                    f"Class {label} occurs in only {label_group_count} groups; at least "
                    f"{minimum_per_class} are needed for group-disjoint test and CV splits."
                )


def choose_group_holdout(
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a group-disjoint holdout with class balance close to the full data."""
    splitter = GroupShuffleSplit(n_splits=200, test_size=test_size, random_state=seed)
    target_positive_rate = float(labels.mean())
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for train_indices, test_indices in splitter.split(labels, labels, groups):
        if len(np.unique(labels[train_indices])) < 2 or len(np.unique(labels[test_indices])) < 2:
            continue
        size_error = abs(len(test_indices) / len(labels) - test_size)
        balance_error = abs(float(labels[test_indices].mean()) - target_positive_rate)
        score = size_error + balance_error
        if best is None or score < best[0]:
            best = (score, train_indices, test_indices)

    if best is None:
        raise ValueError(
            "Could not create a group-disjoint holdout containing both classes. "
            "Check group coverage or provide more recordings."
        )
    return best[1], best[2]


def split_train_test(
    labels: np.ndarray,
    groups: np.ndarray | None,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    if groups is not None:
        return choose_group_holdout(labels, groups, test_size, seed)
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )


def validate_split_for_cross_validation(
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    folds: int,
    groups: np.ndarray | None,
) -> None:
    train_counts = class_counts(labels[train_indices])
    test_counts = class_counts(labels[test_indices])
    if min(train_counts.get(0, 0), train_counts.get(1, 0)) < folds:
        raise ValueError(
            f"The training split needs at least {folds} samples of each class for "
            f"{folds}-fold CV, but class counts are {train_counts}."
        )
    if min(test_counts.get(0, 0), test_counts.get(1, 0)) < 1:
        raise ValueError(f"The test split must contain both classes; got {test_counts}.")
    if groups is not None:
        train_groups = groups[train_indices]
        for label in (0, 1):
            groups_for_label = np.unique(train_groups[labels[train_indices] == label])
            if len(groups_for_label) < folds:
                raise ValueError(
                    f"Class {label} occurs in only {len(groups_for_label)} training "
                    f"groups; at least {folds} are required for group CV."
                )


def iter_folds(
    train_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    folds: int,
    seed: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    train_labels = labels[train_indices]
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        iterator = splitter.split(train_indices, train_labels)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        iterator = splitter.split(
            train_indices, train_labels, groups=groups[train_indices]
        )

    for relative_train, relative_validation in iterator:
        fold_train = train_indices[relative_train]
        fold_validation = train_indices[relative_validation]
        if len(np.unique(labels[fold_validation])) < 2:
            raise ValueError("A validation fold contains only one class.")
        yield fold_train, fold_validation


def fit_standardizer(
    deterministic_features: Sequence[np.ndarray], indices: Sequence[int]
) -> Standardizer:
    feature_dim = deterministic_features[int(indices[0])].shape[1]
    total_count = 0
    total_sum = np.zeros(feature_dim, dtype=np.float64)
    total_squared_sum = np.zeros(feature_dim, dtype=np.float64)

    for index in indices:
        features = deterministic_features[int(index)].astype(np.float64, copy=False)
        total_count += len(features)
        total_sum += features.sum(axis=0)
        total_squared_sum += np.square(features).sum(axis=0)

    mean = total_sum / total_count
    variance = np.maximum(total_squared_sum / total_count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return Standardizer(mean.astype(np.float32), scale.astype(np.float32))


def collate_bags(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, indices = zip(*batch)
    lengths = torch.tensor([len(item) for item in features], dtype=torch.long)
    max_windows = int(lengths.max())
    feature_dim = int(features[0].shape[1])
    padded = torch.zeros((len(batch), max_windows, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_windows), dtype=torch.bool)

    for row, item in enumerate(features):
        padded[row, : len(item)] = item
        mask[row, : len(item)] = True

    return (
        padded,
        mask,
        torch.stack(labels),
        torch.tensor(indices, dtype=torch.long),
    )


def make_loader(
    dataset: BagDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    use_cuda: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_bags,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=seed_data_loader_worker if num_workers > 0 else None,
    )


def seed_data_loader_worker(worker_id: int) -> None:
    """Give every worker an independent, reproducible short-padding RNG."""
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is not None and isinstance(worker_info.dataset, BagDataset):
        worker_info.dataset.rng = np.random.default_rng(worker_seed)


def aggregate_top_k_probabilities(
    window_logits: torch.Tensor, mask: torch.Tensor, top_k: int
) -> torch.Tensor:
    probabilities = torch.sigmoid(window_logits)
    probabilities = probabilities.masked_fill(~mask, float("-inf"))
    selected_k = min(top_k, probabilities.shape[1])
    top_values = torch.topk(probabilities, k=selected_k, dim=1).values
    valid_counts = mask.sum(dim=1).clamp(max=selected_k)
    top_mask = (
        torch.arange(selected_k, device=window_logits.device)[None, :]
        < valid_counts[:, None]
    )
    top_values = torch.where(top_mask, top_values, torch.zeros_like(top_values))
    return top_values.sum(dim=1) / valid_counts


def balanced_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = class_counts(labels)
    total = len(labels)
    weights = [total / (2.0 * counts[label]) for label in (0, 1)]
    return torch.tensor(weights, dtype=torch.float32)


def compute_loss(
    window_logits: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
    class_weights: torch.Tensor,
    negative_instance_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bag_probabilities = aggregate_top_k_probabilities(window_logits, mask, top_k)
    safe_probabilities = bag_probabilities.clamp(1e-6, 1.0 - 1e-6)
    bag_losses = F.binary_cross_entropy(
        safe_probabilities, labels, reduction="none"
    )
    sample_weights = class_weights[labels.long()]
    bag_loss = (bag_losses * sample_weights).mean()

    # A Top-k bag loss only updates the current hardest negative windows. This
    # auxiliary term explicitly pushes every valid window in a negative bag down.
    negative_mask = mask & (labels[:, None] == 0)
    if negative_mask.any():
        negative_logits = window_logits[negative_mask]
        negative_instance_loss = F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )
    else:
        negative_instance_loss = window_logits.sum() * 0.0

    total_loss = bag_loss + negative_instance_weight * negative_instance_loss
    return total_loss, bag_probabilities


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []

    for windows, mask, labels, indices in loader:
        windows = windows.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(windows)
        probabilities = aggregate_top_k_probabilities(logits, mask, top_k)
        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.numpy())
        all_indices.append(indices.numpy())

    return (
        np.concatenate(all_probabilities),
        np.concatenate(all_labels).astype(np.int64),
        np.concatenate(all_indices).astype(np.int64),
    )


def train_fold(
    fold_number: int,
    fold_train: np.ndarray,
    fold_validation: np.ndarray,
    mfcc_samples: Sequence[np.ndarray],
    deterministic_features: Sequence[np.ndarray],
    labels: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    standardizer = fit_standardizer(deterministic_features, fold_train)
    input_dim = int(standardizer.mean.shape[0])
    fold_seed = args.seed + fold_number

    train_dataset = BagDataset(
        fold_train,
        mfcc_samples,
        deterministic_features,
        labels,
        standardizer,
        args.window_size,
        args.hop_size,
        random_short_padding=True,
        seed=fold_seed,
    )
    validation_dataset = BagDataset(
        fold_validation,
        mfcc_samples,
        deterministic_features,
        labels,
        standardizer,
        args.window_size,
        args.hop_size,
        random_short_padding=False,
        seed=fold_seed,
    )
    test_dataset = BagDataset(
        test_indices,
        mfcc_samples,
        deterministic_features,
        labels,
        standardizer,
        args.window_size,
        args.hop_size,
        random_short_padding=False,
        seed=fold_seed,
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=fold_seed,
        use_cuda=device.type == "cuda",
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=fold_seed,
        use_cuda=device.type == "cuda",
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=fold_seed,
        use_cuda=device.type == "cuda",
    )

    model = WindowLogisticRegression(input_dim).to(device)
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
                loss, _ = compute_loss(
                    logits,
                    mask,
                    batch_labels,
                    args.top_k,
                    class_weights,
                    args.negative_instance_weight,
                )
                validation_loss_sum += float(loss.item()) * len(batch_labels)
                validation_items += len(batch_labels)

        validation_loss = validation_loss_sum / validation_items
        print(
            f"fold={fold_number} epoch={epoch:03d} "
            f"validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No model checkpoint was selected.")
    model.load_state_dict(best_state)

    validation_probabilities, _, validation_order = predict(
        model, validation_loader, device, args.top_k
    )
    test_probabilities, _, test_order = predict(model, test_loader, device, args.top_k)
    if not np.array_equal(test_order, test_indices):
        raise RuntimeError("Unexpected test prediction order.")

    checkpoint = {
        "model_state_dict": {key: value.cpu() for key, value in best_state.items()},
        "standardizer_mean": standardizer.mean,
        "standardizer_scale": standardizer.scale,
        "input_dim": input_dim,
        "window_size": args.window_size,
        "hop_size": args.hop_size,
        "top_k": args.top_k,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    return validation_probabilities, test_probabilities, {
        "validation_order": validation_order,
        "checkpoint": checkpoint,
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.01, 0.99, 99)
    scored = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int64)
        scored.append((f1_score(labels, predictions, zero_division=0), threshold))
    best_f1 = max(score for score, _ in scored)
    tied_thresholds = [
        threshold for score, threshold in scored if np.isclose(score, best_f1)
    ]
    return float(min(tied_thresholds, key=lambda value: abs(value - 0.5)))


def evaluate_binary_classifier(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(labels, predictions)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "sample_count": int(len(labels)),
        "class_counts": class_counts(labels),
    }


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

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
            "WARNING: no --groups file was provided. The split is sample-disjoint "
            "but cannot guarantee original-recording disjointness."
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
        iter_folds(
            train_indices, labels, groups, args.folds, args.seed
        ),
        start=1,
    ):
        validation_probabilities, test_probabilities, artifacts = train_fold(
            fold_number,
            fold_train,
            fold_validation,
            mfcc_samples,
            deterministic_features,
            labels,
            test_indices,
            args,
            device,
        )
        validation_order = np.asarray(artifacts["validation_order"], dtype=np.int64)
        out_of_fold_probabilities[validation_order] = validation_probabilities
        fold_test_probabilities.append(test_probabilities)
        checkpoints.append(artifacts["checkpoint"])

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold_number, checkpoint in enumerate(checkpoints, start=1):
        torch.save(checkpoint, args.output_dir / f"fold_{fold_number}.pt")

    np.save(args.output_dir / "train_indices.npy", train_indices)
    np.save(args.output_dir / "test_indices.npy", test_indices)
    np.save(args.output_dir / "oof_probabilities.npy", training_oof_probabilities)
    np.save(args.output_dir / "test_probabilities.npy", test_probabilities)

    results = {
        "configuration": vars(args),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "threshold_source": "training out-of-fold predictions",
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_ready(results), file, ensure_ascii=False, indent=2)

    print("\nTraining OOF metrics:")
    print(json.dumps(json_ready(oof_metrics), ensure_ascii=False, indent=2))
    print("\nHeld-out test metrics:")
    print(json.dumps(json_ready(test_metrics), ensure_ascii=False, indent=2))
    print(f"\nArtifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
