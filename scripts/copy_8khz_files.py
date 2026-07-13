import shutil
from pathlib import Path


def main():
    source_dir = Path("data/raw/ood_positive/Mosquitos Audio Samples")
    target_dir = Path("data/raw/ood_positive/vasconcelos")
    
    target_dir.mkdir(exist_ok=True)
    
    wav_files = list(source_dir.rglob("8khz/*.wav"))
    print(f"Found {len(wav_files)} 8khz WAV files")
    
    copied = 0
    for wav_file in wav_files:
        target_path = target_dir / wav_file.name
        shutil.copy2(wav_file, target_path)
        copied += 1
    
    print(f"Copied {copied} files to {target_dir}")
    
    total_wav = len(list(target_dir.glob("*.wav")))
    print(f"Total WAV files in vasconcelos/: {total_wav}")


if __name__ == "__main__":
    main()