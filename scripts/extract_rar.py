from pyunpack import Archive
from pathlib import Path


def main():
    rar_path = Path("data/raw/ood_positive/Mosquitos Audio Samples.rar")
    output_dir = Path("data/raw/ood_positive")
    
    print(f"Extracting {rar_path} to {output_dir}...")
    
    Archive(str(rar_path)).extractall(str(output_dir))
    
    print("Extraction completed!")
    
    vasconcelos_dir = output_dir / "vasconcelos"
    if vasconcelos_dir.exists():
        wav_count = len(list(vasconcelos_dir.glob("*.wav")))
        print(f"Found {wav_count} WAV files in vasconcelos/")
    else:
        print("Checking for extracted files...")
        for item in output_dir.iterdir():
            if item.is_dir():
                wav_count = len(list(item.glob("*.wav")))
                print(f"  {item.name}: {wav_count} WAV files")


if __name__ == "__main__":
    main()