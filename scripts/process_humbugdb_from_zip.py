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
        help="Maximum number of audio files to process (for testing)"
    )
    return parser.parse_args()


def load_metadata(csv_path: Path) -> Dict[str, List[dict]]:
    """Load metadata CSV and group by recording name."""
    recordings: Dict[str, List[dict]] = {}
    
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if name not in recordings:
                recordings[name] = []
            recordings[name].append(row)
    
    return recordings


def find_zip_containing_id(zip_dir: Path, record_id: str) -> Path | None:
    """Find which zip file contains the audio with given id."""
    zip_filename = f"{record_id}.wav"
    for zip_path in sorted(zip_dir.glob("humbugdb_neurips_2021_*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if zip_filename in zf.namelist():
                    return zip_path
        except zipfile.BadZipFile:
            continue
    return None


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


def process_recording(
    zip_path: Path,
    record_id: str,
    segments: List[dict],
    target_sr: int,
    n_mfcc: int,
    hop_length: int,
    n_fft: int,
    window_size: int,  
    min_duration: float,
    max_duration: float  # 【新增】传入最大时长参数
) -> List[Dict]:
    """Process a single recording and extract features for each labeled segment."""
    y, sr = load_audio_from_zip(zip_path, record_id, target_sr)
    results = []
    
    for segment in segments:
        sound_type = segment["sound_type"].strip().lower()
        
        if sound_type not in ("mosquito", "background", "audio"):
            continue
        
        duration = float(segment["length"])
        
        # 【修改】同时判断最小和最大时长
        if duration < min_duration or duration > max_duration:
            continue
        
        actual_sr = int(segment["sample_rate"])
        
        if actual_sr != sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=actual_sr)
            sr_used = actual_sr
        else:
            y_resampled = y
            sr_used = sr
        
        samples_needed = int(duration * sr_used)
        if len(y_resampled) >= samples_needed:
            y_segment = y_resampled[:samples_needed]
        else:
            y_segment = np.pad(y_resampled, (0, max(0, samples_needed - len(y_resampled))))
        
        mfcc_features = extract_mfcc(y_segment, sr_used, n_mfcc, hop_length, n_fft, window_size)
        
        if mfcc_features is None:
            continue
        
        results.append({
            "record_id": record_id,
            "name": segment["name"],
            "sound_type": sound_type,
            "length": duration,
            "sample_rate": sr_used,
            "mfcc_shape": mfcc_features.shape,
            "mfcc": mfcc_features
        })
    
    return results


def main() -> None:
    args = parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metadata from {args.csv}")
    recordings = load_metadata(args.csv)
    print(f"Found {len(recordings)} unique recordings")
    
    all_features = []
    processed = 0
    skipped = 0
    skipped_filtered = 0  # 【修改】改名，涵盖太短和太长的情况
    
    for name, segments in recordings.items():
        if args.max_files and processed >= args.max_files:
            break
        
        record_id = segments[0]["id"]
        zip_path = find_zip_containing_id(args.zip_dir, record_id)
        
        if zip_path is None:
            skipped += 1
            continue
        
        print(f"Processing {name} (id={record_id})...")
        
        try:
            features = process_recording(
                zip_path, record_id, segments,
                args.sample_rate, args.n_mfcc, args.hop_length, args.n_fft,
                args.window_size, args.min_duration, args.max_duration  # 【修改】传入新参数
            )
            all_features.extend(features)
            processed += 1
            if len(features) == 0:
                skipped_filtered += 1
        except Exception as e:
            print(f"Error processing {name}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessed {processed} recordings, skipped {skipped}, skipped/filtered {skipped_filtered}")
    print(f"Extracted {len(all_features)} features")
    
    if not all_features:
        print("No features extracted. Exiting.")
        return
    
    labels = [1 if f["sound_type"] == "mosquito" else 0 for f in all_features]
    
    mfcc_list = [f["mfcc"] for f in all_features]
    mfcc_array = np.empty(len(mfcc_list), dtype=object)
    mfcc_array[:] = mfcc_list
    
    print(f"\nFeature array shape (object): {mfcc_array.shape}")
    print(f"Sample feature shapes: {[f.shape for f in mfcc_array[:3]]}")
    print(f"Labels: {sum(labels)} mosquito, {len(labels) - sum(labels)} non-mosquito")
    
    np.save(args.output / "mfcc_features.npy", mfcc_array)
    np.save(args.output / "labels.npy", labels)
    
    metadata = {
        "n_mfcc": args.n_mfcc,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "window_size": args.window_size,  
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,  # 【新增】记录到 metadata
        "total_segments": len(all_features),
        "mosquito_count": sum(labels),
        "non_mosquito_count": len(labels) - sum(labels),
        "processed_recordings": processed,
        "skipped_recordings": skipped,
        "skipped_filtered_segments": skipped_filtered  # 【修改】更新字段名
    }
    with (args.output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved features to {args.output}")
    print(f"  - mfcc_features.npy: {len(mfcc_array)} variable-length features (dtype=object, padded to multiples of {args.window_size})")
    print(f"  - labels.npy: {len(labels)} labels")
    print(f"  - metadata.json: processing parameters")


if __name__ == "__main__":
    main()