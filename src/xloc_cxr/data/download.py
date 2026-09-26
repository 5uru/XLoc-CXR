"""Download VinDr-CXR (VinBigData Kaggle competition) dataset."""
import os
import shutil
import subprocess
import zipfile
from multiprocessing import Pool
from pathlib import Path

COMPETITION = "vinbigdata-chest-xray-abnormalities-detection"


def _extract_worker(args):
    """Extract a subset of zip entries (one process per chunk)."""
    zip_path, dest, names = args
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in names:
            zf.extract(name, dest)
    return len(names)


def _extract_zip_parallel(zip_path: Path, dest: Path, workers: int = 16) -> None:
    """Parallel ZIP extraction using a process pool.

    ZIP stores each file with independent deflate compression, so
    extraction parallelizes perfectly across entries.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    workers = min(workers, os.cpu_count() or 4, len(names))
    chunks = [names[i::workers] for i in range(workers)]

    print(f"[download] Extracting {len(names)} files with {workers} workers...")
    args = [(str(zip_path), str(dest), chunk) for chunk in chunks]

    with Pool(workers) as pool:
        for done in pool.imap_unordered(_extract_worker, args):
            pass


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract ZIP with the fastest available method."""
    # 7z: multi-threaded ZIP extraction — fastest if installed
    if shutil.which("7z"):
        print("[download] Extracting with 7z...")
        subprocess.run(["7z", "x", str(zip_path), f"-o{dest}", "-y"], check=True)
        return

    # Parallel Python extraction (no system deps needed)
    _extract_zip_parallel(zip_path, dest)


def download_vindr(root: str = "data/vindr_cxr") -> Path:
    """Download dataset from Kaggle if not already present.

    Requires ~/.kaggle/kaggle.json credentials.
    """
    root_path = Path(root)

    if (root_path / "train.csv").exists():
        print(f"[download] Dataset already present at {root_path}")
        return root_path

    root_path.mkdir(parents=True, exist_ok=True)
    zip_path = root_path / f"{COMPETITION}.zip"

    # Skip download if the zip is already there (e.g. after a failed extraction)
    if not zip_path.exists():
        print(f"[download] Downloading '{COMPETITION}' → {root_path}")

        from kaggle.api.kaggle_api_extended import KaggleApi
        from requests.exceptions import HTTPError

        api = KaggleApi()
        try:
            api.authenticate()
        except OSError as e:
            raise RuntimeError(
                "Kaggle credentials not found.\n"
                "  1. Go to https://www.kaggle.com/settings → 'Create New API Token'\n"
                "  2. Copy kaggle.json to /root/.kaggle/kaggle.json\n"
                "  3. chmod 600 /root/.kaggle/kaggle.json"
            ) from e

        try:
            api.competition_download_files(COMPETITION, path=str(root_path), quiet=False)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                raise RuntimeError(
                    "403 Forbidden — you must accept the competition rules first:\n"
                    "  1. Go to https://www.kaggle.com/competitions/"
                    "vinbigdata-chest-xray-abnormalities-detection/rules\n"
                    "  2. Click 'I Understand and Accept'\n"
                    "  (If the button is missing, verify your phone number in Kaggle settings)\n"
                    "  3. Re-run this command"
                ) from e
            raise
    else:
        print(f"[download] Zip already downloaded: {zip_path} — extracting")

    if zip_path.exists():
        _extract_zip(zip_path, root_path)
        zip_path.unlink()

    if not (root_path / "train.csv").exists():
        raise RuntimeError(f"Download succeeded but train.csv not found in {root_path}")

    print("[download] Done")
    return root_path
