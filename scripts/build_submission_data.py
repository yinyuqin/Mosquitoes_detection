from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "jupyter-notebook" / "submission_data.zip"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def wav_bytes(y: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, y, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def mfcc39(y: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 8000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sample_rate, target_sr=8000)
        sample_rate = 8000
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13, n_fft=512, hop_length=256)
    if mfcc.shape[1] >= 9:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
    else:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    return np.concatenate([mfcc, delta, delta2], axis=0).astype(np.float32)


def read_audio_member(archive: Path, predicate) -> tuple[np.ndarray, int, str]:
    with zipfile.ZipFile(archive) as source:
        member = next(name for name in source.namelist() if predicate(Path(name)))
        data = source.read(member)
    y, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32), sample_rate, member


def find_humbug_record(record_id: str) -> tuple[np.ndarray, int, str]:
    for archive in sorted((ROOT / "data/raw/humbugdb").glob("humbugdb_neurips_2021_*.zip")):
        try:
            return read_audio_member(archive, lambda path: path.stem == record_id)
        except StopIteration:
            continue
    raise FileNotFoundError(record_id)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    def add_bytes(target: zipfile.ZipFile, arcname: str, data: bytes, source: str) -> None:
        target.writestr(arcname, data, compress_type=zipfile.ZIP_DEFLATED)
        entries.append({"path": arcname, "bytes": len(data), "sha256": sha256_bytes(data), "source": source})

    def add_file(target: zipfile.ZipFile, path: Path, arcname: str) -> None:
        data = path.read_bytes()
        add_bytes(target, arcname, data, str(path.relative_to(ROOT)).replace("\\", "/"))

    with zipfile.ZipFile(OUTPUT, "w", allowZip64=True) as target:
        ood_sources = {
            "data1.zip": ROOT / "data/raw/ood_negative/data1.zip",
            "data2.zip": ROOT / "data/raw/ood_negative/data2.zip",
            "vasconcelos.zip": ROOT / "data/raw/ood_positive/vasconcelos.zip",
            "other.zip": ROOT / "data/raw/ood_positive/other.zip",
        }
        for name, path in ood_sources.items():
            add_file(target, path, f"submission_data/ood/{name}")

        logistic_dir = ROOT / "outputs/logistic_mil_gpu"
        for name in [
            *(f"fold_{fold}.pt" for fold in range(1, 6)),
            "train_indices.npy", "test_indices.npy", "oof_probabilities.npy",
            "test_probabilities.npy", "metrics.json",
        ]:
            add_file(target, logistic_dir / name, f"submission_data/logistic_mil/{name}")

        processed_dir = ROOT / "data/processed/humbugdb_mfcc"
        for name in ["metadata.json", "labels.npy", "groups.npy", "sample_ids.npy"]:
            add_file(target, processed_dir / name, f"submission_data/dataset/{name}")

        model_dirs = [
            "logistic_global_gpu", "mlp_global_gpu", "mlp_global_gpu_grouped", "mlp_mil_gpu"
        ]
        for directory in model_dirs:
            for name in ["metrics.json", "test_indices.npy", "test_probabilities.npy", "oof_probabilities.npy"]:
                path = ROOT / "outputs" / directory / name
                if path.exists():
                    add_file(target, path, f"submission_data/comparison/id/{directory}/{name}")
        for name in ["metrics.json", "probabilities.npz", "predictions.csv"]:
            add_file(
                target,
                ROOT / "outputs/ood_evaluation_balanced" / name,
                f"submission_data/comparison/ood_balanced/{name}",
            )

        labels = np.load(ROOT / "data/processed/humbugdb_mfcc/labels.npy")
        ids = np.load(ROOT / "data/processed/humbugdb_mfcc/sample_ids.npy", allow_pickle=True)
        features = np.load(ROOT / "data/processed/humbugdb_mfcc/mfcc_features.npy", allow_pickle=True)
        example_specs = [
            ("id_mosquito", str(ids[np.flatnonzero(labels == 1)[0]]), int(np.flatnonzero(labels == 1)[0])),
            ("id_background", str(ids[np.flatnonzero(labels == 0)[0]]), int(np.flatnonzero(labels == 0)[0])),
        ]
        for name, record_id, index in example_specs:
            y, sample_rate, member = find_humbug_record(record_id)
            y = y[: sample_rate * 5]
            if sample_rate != 8000:
                y = librosa.resample(y, orig_sr=sample_rate, target_sr=8000)
                sample_rate = 8000
            add_bytes(target, f"submission_data/examples/{name}.wav", wav_bytes(y, sample_rate), member)
            add_bytes(target, f"submission_data/examples/{name}_mfcc.npy", npy_bytes(np.asarray(features[index], dtype=np.float32)), "processed HumBugDB feature")

        ood_examples = [
            ("ood_mosquito", ROOT / "data/raw/ood_positive/other.zip"),
            ("ood_background", ROOT / "data/raw/ood_negative/data1.zip"),
        ]
        for name, archive in ood_examples:
            y, sample_rate, member = read_audio_member(archive, lambda path: path.suffix.lower() == ".wav")
            y = y[: sample_rate * 5]
            if sample_rate != 8000:
                y = librosa.resample(y, orig_sr=sample_rate, target_sr=8000)
                sample_rate = 8000
            add_bytes(target, f"submission_data/examples/{name}.wav", wav_bytes(y, sample_rate), member)
            add_bytes(target, f"submission_data/examples/{name}_mfcc.npy", npy_bytes(mfcc39(y, sample_rate)), member)

        manifest = {
            "schema_version": 1,
            "purpose": "small reproducibility bundle for mosquito_detection.ipynb",
            "seed": 2026,
            "not_included": [
                "HumBugDB raw archives (~4.06 GB)",
                "full processed HumBugDB MFCC array (~588 MB)",
            ],
            "official_humbugdb": "https://zenodo.org/records/4904800",
            "ood_protocol": {
                "full_valid_counts": {"negative": 170, "positive": 1699},
                "ppt_balanced_counts": {"negative": 170, "positive": 170},
                "selection_seed": 2026,
                "threshold_tuning": "forbidden; reuse training OOF threshold",
            },
            "entries": entries,
        }
        target.writestr(
            "submission_data/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024**2:.1f} MiB, {len(entries)} payload files)")


if __name__ == "__main__":
    main()
