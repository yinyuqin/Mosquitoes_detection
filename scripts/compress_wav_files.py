from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress WAV files into zip archives and remove originals."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Root directory to search for WAV files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without actually compressing or deleting"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing zip files"
    )
    return parser.parse_args()


def compress_directory(dir_path: Path, dry_run: bool = False, force: bool = False) -> bool:
    """Compress all WAV files in a directory into a zip archive."""
    wav_files = sorted(dir_path.glob("*.wav"))
    
    if not wav_files:
        print(f"  No WAV files found in {dir_path.name}, skipping")
        return False
    
    zip_name = dir_path.name + ".zip"
    zip_path = dir_path.parent / zip_name
    
    if zip_path.exists():
        if force:
            print(f"  Warning: {zip_name} already exists, will overwrite")
        else:
            print(f"  {zip_name} already exists, skipping (use --force to overwrite)")
            return False
    
    if dry_run:
        print(f"  [DRY RUN] Would create {zip_name} with {len(wav_files)} files")
        return True
    
    print(f"  Creating {zip_name} with {len(wav_files)} files...")
    
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for wav_file in wav_files:
                arcname = wav_file.name
                zf.write(wav_file, arcname=arcname)
        
        if zip_path.exists() and zip_path.stat().st_size > 0:
            print(f"  ✓ Zip created successfully")
            return True
        else:
            print(f"  ✗ Zip creation failed or empty")
            return False
    except Exception as e:
        print(f"  ✗ Error creating zip: {e}")
        return False


def verify_zip_integrity(zip_path: Path, expected_count: int) -> bool:
    """Verify that the zip file is valid and contains expected files."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            corrupted = zf.testzip()
            actual_count = len(zf.namelist())
            
            if corrupted:
                print(f"  ✗ Zip integrity check failed: corrupted file {corrupted}")
                return False
            
            if actual_count != expected_count:
                print(f"  ✗ Zip file count mismatch: expected {expected_count}, got {actual_count}")
                return False
            
            print(f"  ✓ Zip integrity verified: {actual_count} files, no corruption")
            return True
    except Exception as e:
        print(f"  ✗ Error verifying zip: {e}")
        return False


def delete_wav_files(dir_path: Path, dry_run: bool = False) -> int:
    """Delete all WAV files in a directory."""
    wav_files = sorted(dir_path.glob("*.wav"))
    
    if dry_run:
        print(f"  [DRY RUN] Would delete {len(wav_files)} WAV files")
        return len(wav_files)
    
    deleted = 0
    for wav_file in wav_files:
        try:
            os.remove(wav_file)
            deleted += 1
        except Exception as e:
            print(f"    Failed to delete {wav_file.name}: {e}")
    
    print(f"  ✓ Deleted {deleted} WAV files")
    return deleted


def cleanup_empty_directory(dir_path: Path, dry_run: bool = False) -> bool:
    """Remove empty directory if all files have been deleted."""
    remaining_files = list(dir_path.glob("*"))
    
    if not remaining_files:
        if dry_run:
            print(f"  [DRY RUN] Would remove empty directory {dir_path.name}")
            return True
        try:
            shutil.rmtree(dir_path)
            print(f"  ✓ Removed empty directory {dir_path.name}")
            return True
        except Exception as e:
            print(f"  ✗ Failed to remove directory {dir_path.name}: {e}")
            return False
    else:
        print(f"  Directory {dir_path.name} still has {len(remaining_files)} files, keeping")
        return False


def main() -> None:
    args = parse_args()
    
    directories_to_compress = [
        args.data_dir / "ood_negative" / "data1",
        args.data_dir / "ood_negative" / "data2",
        args.data_dir / "ood_positive" / "other",
        args.data_dir / "ood_positive" / "vasconcelos",
    ]
    
    print(f"Found {len(directories_to_compress)} directories to process")
    print("-" * 60)
    
    total_compressed = 0
    total_deleted = 0
    total_cleaned = 0
    
    for dir_path in directories_to_compress:
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist, skipping")
            continue
        
        print(f"\nProcessing {dir_path.parent.name}/{dir_path.name}")
        
        wav_files = sorted(dir_path.glob("*.wav"))
        num_wav_files = len(wav_files)
        
        success = compress_directory(dir_path, args.dry_run, args.force)
        
        if success and not args.dry_run:
            zip_path = dir_path.parent / (dir_path.name + ".zip")
            if verify_zip_integrity(zip_path, num_wav_files):
                deleted = delete_wav_files(dir_path, args.dry_run)
                total_deleted += deleted
                
                cleaned = cleanup_empty_directory(dir_path, args.dry_run)
                if cleaned:
                    total_cleaned += 1
                
                total_compressed += 1
            else:
                print(f"  ✗ Zip integrity check failed, NOT deleting WAV files")
    
    print("\n" + "-" * 60)
    print("Summary:")
    print(f"  Directories compressed: {total_compressed}")
    print(f"  WAV files deleted: {total_deleted}")
    print(f"  Empty directories removed: {total_cleaned}")
    
    if args.dry_run:
        print("\nNote: This was a dry run. No files were actually created or deleted.")


if __name__ == "__main__":
    main()