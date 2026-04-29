"""
TalkHand BISINDO - Data Acquisition Script
Mengambil dataset dari 2 sumber utama: GitHub dan Local
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"

DATASET_SOURCES = {
    "github_bisindo": {
        "source": "GitHub",
        "url": "https://raw.githubusercontent.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset/main",
        "method": "git clone / ZIP download via requests",
        "description": "Dataset BISINDO hand sign detection dari komunitas lokal Indonesia",
        "estimated_labels": 26,
        "format": "Image dataset dengan struktur folder per label"
    },
    "local_citra_bisindo": {
        "source": "Local",
        "url": str(BASE_DIR / "Citra BISINDO"),
        "method": "Direct folder access",
        "description": "Local dataset Citra BISINDO with images per letter",
        "estimated_labels": 26,
        "format": "Image folders per class"
    }
}

BISINDO_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def print_acquisition_plan():
    print("=" * 65)
    print("  TALKHAND BISINDO - DATA ACQUISITION PLAN")
    print("=" * 65)
    for key, src in DATASET_SOURCES.items():
        print(f"\n[Source] [{src['source']}] {key}")
        print(f"   URL      : {src['url']}")
        print(f"   Metode   : {src['method']}")
        print(f"   Format   : {src['format']}")
        print(f"   Labels   : ~{src['estimated_labels']} isyarat")
        print(f"   Deskripsi: {src['description']}")

def create_dataset_structure():
    """Buat struktur folder untuk setiap dataset source."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    for key, src in DATASET_SOURCES.items():
        if src["source"].lower() == "local":
            dest = DATA_RAW / key
            src_folder = BASE_DIR / "Citra BISINDO"
            if src_folder.exists():
                if dest.exists():
                    print(f"[INFO] Local dataset already present: {dest}")
                else:
                    shutil.copytree(src_folder, dest)
                    print(f"[OK] Local dataset copied to: {dest}")
            else:
                print(f"[WARNING] Local source folder tidak ditemukan: {src_folder}")
                dest.mkdir(parents=True, exist_ok=True)
        else:
            for split in ["train", "val", "test"]:
                for letter in BISINDO_ALPHABET:
                    folder = DATA_RAW / key / split / letter
                    folder.mkdir(parents=True, exist_ok=True)

    # Simpan metadata
    meta = {
        "project": "TalkHand BISINDO",
        "dataset_sources": DATASET_SOURCES,
        "labels": BISINDO_ALPHABET,
        "total_labels": len(BISINDO_ALPHABET),
        "splits": ["train", "val", "test"],
        "split_ratio": "70% / 15% / 15%"
    }
    with open(DATA_RAW / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    external_sources = [k for k, v in DATASET_SOURCES.items() if v["source"].lower() != "local"]
    print(f"\n[OK] Struktur folder dataset dibuat ({len(BISINDO_ALPHABET)} label × 3 split × {len(external_sources)} external source + local)")

def print_download_commands():
    print("\n" + "=" * 65)
    print("  CARA DOWNLOAD DATASET")
    print("=" * 65)
    print("""
1. GITHUB (BISINDO Hand Sign Detection Dataset):
   $ git clone https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset
   # atau via Python:
   >>> import urllib.request, zipfile
   >>> url = "https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset/archive/refs/heads/main.zip"
   >>> urllib.request.urlretrieve(url, "bisindo_github.zip")
   >>> with zipfile.ZipFile("bisindo_github.zip") as z:
   ...     z.extractall("data/raw/github_bisindo/")

2. LOCAL (Citra BISINDO):
   No download needed. Dataset already copied to 'data/raw/local_citra_bisindo/'
""")

if __name__ == "__main__":
    print_acquisition_plan()
    create_dataset_structure()
    print_download_commands()
    print(f"\nLabels BISINDO ({len(BISINDO_ALPHABET)}): {BISINDO_ALPHABET}")
    print(f"\nTarget dataset awal: 200-500 sampel per label")
    print(f"   Total estimasi: {200 * len(BISINDO_ALPHABET):,} - {500 * len(BISINDO_ALPHABET):,} gambar\n")
