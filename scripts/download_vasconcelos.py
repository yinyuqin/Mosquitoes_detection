import requests
import os
from pathlib import Path


def download_vasconcelos_dataset():
    article_id = 11902125
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    
    print("Fetching article metadata from Figshare API...")
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Failed to fetch metadata: {response.status_code}")
        return False
    
    data = response.json()
    print(f"Title: {data.get('title')}")
    
    files = data.get("files", [])
    if not files:
        print("No files found")
        return False
    
    download_url = files[0]["download_url"]
    file_name = files[0]["name"]
    file_size = files[0]["size"]
    
    print(f"Downloading {file_name} ({file_size / 1024 / 1024:.2f} MB)...")
    
    output_dir = Path("data/raw/ood_positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    download_path = output_dir / file_name
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    response = requests.get(download_url, headers=headers, stream=True)
    if response.status_code != 200:
        print(f"Download failed: {response.status_code}")
        return False
    
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    
    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end="")
    
    print(f"\nDownloaded to: {download_path}")
    
    if file_name.endswith(".rar"):
        print("\nNote: The file is a RAR archive.")
        print("You need to extract it manually using WinRAR or 7-Zip.")
        print(f"RAR file location: {download_path}")
        print("Extract the contents to: data/raw/ood_positive/vasconcelos/")
    else:
        print("Unexpected file format")
    
    return True


if __name__ == "__main__":
    success = download_vasconcelos_dataset()
    if success:
        print("\n✓ Download completed!")
    else:
        print("\n✗ Failed to download dataset")