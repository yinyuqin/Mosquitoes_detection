from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP_DIR = ROOT / "data" / "raw" / "humbugdb"
DEFAULT_CSV = ROOT / "data" / "raw" / "humbugdb" / "neurips_2021_zenodo_0_0_1.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "humbugdb_mfcc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process HumBugDB audio from zip files and extract MFCC features."
    )
    parser.add_argument(
        "--zip-dir", type=Path, default=DEFAULT_ZIP_DIR,
        help="Directory containing humbugdb zip files"
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help="Path to neurips_2021_zenodo_0_0_1.csv metadata file"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for processed MFCC features"
    )
    parser.add_argument(
        "--n-mfcc", type=int, default=13,
        help="Number of MFCC coefficients to extract"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=8000,
        help="Target sample rate for audio resampling"
    )
    parser.add_argument(
        "--hop-length", type=int, default=256,
        help="Hop length for STFT"
    )
    parser.add_argument(
        "--n-fft", type=int, default=512,
        help="FFT window size"
    )
    parser.add_argument(
        "--window-size", type=int, default=64,
        help="Base window size for MFCC frames. Features will be padded to a multiple of this size."
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.1,
        help="Minimum duration in seconds for valid audio segments"
    )
    # 【新增】最大时长限制参数
    parser.add_argument(
        "--max-duration", type=float, default=60.0,
        help="Maximum duration in seconds for valid audio segments (discard if longer)"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of labelled audio clips to process (for testing)"
    )
    return parser.parse_args()


def load_metadata(csv_path: Path) -> List[dict]:
    """Load one metadata row per labelled audio clip."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_zip_index(zip_dir: Path) -> Dict[str, Path]:
    """Map each clip ID to its ZIP once instead of rescanning ZIPs per sample."""
    index: Dict[str, Path] = {}
    for zip_path in sorted(zip_dir.glob("humbugdb_neurips_2021_*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    member_path = Path(member)
                    if member_path.suffix.lower() == ".wav":
                        index[member_path.stem] = zip_path
        except zipfile.BadZipFile:
            continue
    return index


def load_audio_from_zip(zip_path: Path, record_id: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load audio from inside a zip file without extracting."""
    import io 
    zip_filename = f"{record_id}.wav"
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(zip_filename) as audio_file:
            audio_bytes = audio_file.read()
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr)
            return y, sr


def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    hop_length: int = 256,
    n_fft: int = 512,
    window_size: int = 64  
) -> np.ndarray | None:
    """Extract MFCC features from audio signal.
    
    Returns None if the audio is too short to extract valid features.
    Pads features to the nearest multiple of window_size.
    """
    n_samples = len(y)
    min_samples = n_fft
    
    if n_samples < min_samples:
        return None
    
    n_frames = 1 + (n_samples - n_fft) // hop_length
    
    if n_frames < 2:
        return None
    
    actual_n_fft = min(n_fft, n_samples)
    actual_hop = min(hop_length, n_samples - 1)
    
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=actual_hop,
        n_fft=actual_n_fft
    )
    
    n_frames = mfcc.shape[1]
    
    try:
        if n_frames >= 9:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        else:
            mfcc_delta = np.zeros_like(mfcc)
            mfcc_delta2 = np.zeros_like(mfcc)
    except Exception:
        mfcc_delta = np.zeros_like(mfcc)
        mfcc_delta2 = np.zeros_like(mfcc)
    
    features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    actual_frames = features.shape[1]
    if actual_frames % window_size != 0:
        target_frames = math.ceil(actual_frames / window_size) * window_size
        pad_width = target_frames - actual_frames
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    
    return features


def process_segment(
    zip_path: Path,
    segment: dict,
    target_sr: int,
    n_mfcc: int,
    hop_length: int,
    n_fft: int,
    window_size: int,
) -> Dict:
    """Load the clip named by this CSV row and extract one MFCC sample."""
    record_id = segment["id"]
    y, sr = load_audio_from_zip(zip_path, record_id, target_sr)
    if sr != target_sr:
        raise RuntimeError(f"Expected sample rate {target_sr}, got {sr} for {record_id}.")

    duration = float(segment["length"])
    samples_needed = int(round(duration * target_sr))
    if len(y) >= samples_needed:
        y_segment = y[:samples_needed]
    else:
        y_segment = np.pad(y, (0, samples_needed - len(y)))

    mfcc_features = extract_mfcc(
        y_segment, target_sr, n_mfcc, hop_length, n_fft, window_size
    )
    if mfcc_features is None:
        raise ValueError("Audio is too short for MFCC extraction.")

    return {
        "record_id": record_id,
        "name": segment["name"],
        "sound_type": segment["sound_type"].strip().lower(),
        "length": duration,
        "sample_rate": target_sr,
        "mfcc_shape": mfcc_features.shape,
        "mfcc": mfcc_features,
    }


def main() -> None:
    args = parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metadata from {args.csv}")
    segments = load_metadata(args.csv)
    unique_source_recordings = len({row["name"] for row in segments})
    print(
        f"Found {len(segments)} labelled clips from "
        f"{unique_source_recordings} source recordings"
    )

    print(f"Indexing ZIP files in {args.zip_dir}")
    zip_index = build_zip_index(args.zip_dir)
    print(f"Indexed {len(zip_index)} WAV clips")
    
    all_features = []
    processed = 0
    skipped = 0
    skipped_filtered = 0  # 【修改】改名，涵盖太短和太长的情况
    skipped_too_short = 0
    skipped_too_long = 0
    skipped_unsupported_label = 0
    
    for segment in segments:
        if args.max_files and processed >= args.max_files:
            break

        sound_type = segment["sound_type"].strip().lower()
        duration = float(segment["length"])
        if sound_type not in ("mosquito", "background", "audio"):
            skipped_unsupported_label += 1
            skipped_filtered += 1
            continue
        if duration < args.min_duration:
            skipped_too_short += 1
            skipped_filtered += 1
            continue
        if duration > args.max_duration:
            skipped_too_long += 1
            skipped_filtered += 1
            continue

        record_id = segment["id"]
        zip_path = zip_index.get(record_id)
        if zip_path is None:
            skipped += 1
            continue

        try:
            feature = process_segment(
                zip_path,
                segment,
                args.sample_rate,
                args.n_mfcc,
                args.hop_length,
                args.n_fft,
                args.window_size,
            )
            all_features.append(feature)
            processed += 1
            if processed == 1 or processed % 100 == 0:
                print(
                    f"Processed {processed} clips "
                    f"(filtered={skipped_filtered}, missing/failed={skipped})"
                )
        except Exception as e:
            print(f"Error processing id={record_id}: {e}")
            skipped += 1
            continue

    print(
        f"\nProcessed {processed} clips, skipped {skipped}, "
        f"filtered {skipped_filtered}"
    )
    print(f"Extracted {len(all_features)} features")
    
    if not all_features:
        print("No features extracted. Exiting.")
        return
    
    labels = [1 if f["sound_type"] == "mosquito" else 0 for f in all_features]
    groups = np.asarray([f["name"] for f in all_features], dtype=str)
    sample_ids = np.asarray([f["record_id"] for f in all_features], dtype=str)
    
    mfcc_list = [f["mfcc"] for f in all_features]
    mfcc_array = np.empty(len(mfcc_list), dtype=object)
    mfcc_array[:] = mfcc_list
    
    print(f"\nFeature array shape (object): {mfcc_array.shape}")
    print(f"Sample feature shapes: {[f.shape for f in mfcc_array[:3]]}")
    print(f"Labels: {sum(labels)} mosquito, {len(labels) - sum(labels)} non-mosquito")
    
    np.save(args.output / "mfcc_features.npy", mfcc_array)
    np.save(args.output / "labels.npy", labels)
    np.save(args.output / "groups.npy", groups)
    np.save(args.output / "sample_ids.npy", sample_ids)
    
    metadata = {
        "n_mfcc": args.n_mfcc,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "window_size": args.window_size,  
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,  # 【新增】记录到 metadata
        "total_input_rows": len(segments),
        "total_segments": len(all_features),
        "mosquito_count": sum(labels),
        "non_mosquito_count": len(labels) - sum(labels),
        "processed_audio_clips": processed,
        "processed_recordings": len(set(groups.tolist())),
        "skipped_missing_or_failed": skipped,
        "skipped_filtered_segments": skipped_filtered,
        "skipped_too_short": skipped_too_short,
        "skipped_too_long": skipped_too_long,
        "skipped_unsupported_label": skipped_unsupported_label,
    }
    with (args.output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved features to {args.output}")
    print(f"  - mfcc_features.npy: {len(mfcc_array)} variable-length features (dtype=object, padded to multiples of {args.window_size})")
    print(f"  - labels.npy: {len(labels)} labels")
    print(f"  - groups.npy: {len(groups)} source-recording group IDs")
    print(f"  - sample_ids.npy: {len(sample_ids)} clip IDs")
    print(f"  - metadata.json: processing parameters")


if __name__ == "__main__":
    main()
