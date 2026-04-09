"""
Download MovieLens 25M dataset from GroupLens if not already present.

The dataset is ~250 MB zipped. Files land in data/raw/ml-25m/.
We only keep ratings.csv and movies.csv — the rest are discarded to save space.
"""

import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATASET_DIR = RAW_DIR / "ml-25m"
ZIP_PATH = RAW_DIR / "ml-25m.zip"

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

REQUIRED_FILES = ["ratings.csv", "movies.csv"]


def download_with_progress(url: str, dest: Path) -> None:
    """Stream-download a file, showing a tqdm progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 64  # 64 KB chunks

    print(f"Downloading {url}")
    with (
        open(dest, "wb") as f,
        tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def extract_required_files(zip_path: Path, dest_dir: Path) -> None:
    """
    Unzip only ratings.csv and movies.csv from the archive.
    Skips the genome files and other extras we don't need.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if filename in REQUIRED_FILES:
                print(f"Extracting {member} …")
                # Flatten into dest_dir (strip the ml-25m/ prefix)
                data = zf.read(member)
                out_path = dest_dir / filename
                out_path.write_bytes(data)
    print(f"Extracted to {dest_dir}")


def already_downloaded() -> bool:
    """Return True if both required CSVs already exist."""
    return all((DATASET_DIR / f).exists() for f in REQUIRED_FILES)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if already_downloaded():
        print("MovieLens 25M already present — skipping download.")
        for f in REQUIRED_FILES:
            path = DATASET_DIR / f
            size_mb = path.stat().st_size / 1e6
            print(f"  {path.name}: {size_mb:.1f} MB")
        return

    # Download zip if needed
    if not ZIP_PATH.exists():
        download_with_progress(DOWNLOAD_URL, ZIP_PATH)
    else:
        print(f"Zip already present at {ZIP_PATH}, skipping download.")

    # Extract only what we need
    extract_required_files(ZIP_PATH, DATASET_DIR)

    # Clean up the zip to save ~250 MB
    ZIP_PATH.unlink()
    print("Zip removed. Download complete.")


if __name__ == "__main__":
    main()
