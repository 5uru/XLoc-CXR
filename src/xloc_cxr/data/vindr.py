"""VinDr-CXR (VinBigData) dataset loader with on-disk preprocessing cache.

Data layout (Kaggle competition):
  root/
    train.csv          — image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max
    train/             — DICOM images (.dicom)

Cache layout (built once, reused across runs):
  cache_dir/
    <image_id>.npy     — uint8 grayscale image, 224x224
    dims.csv           — image_id, orig_height, orig_width

15 classes (class_id 0-14), 14 pathologies + "No finding" (class_id 14).
An image can have MULTIPLE findings: labels are a multi-hot vector,
bboxes is a dict with one entry per present class.
"""
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from .download import download_vindr
from .transforms import augment_normalize, MEAN, STD

# Class id ↔ name mapping (from competition page)
CLASSES = [
    "Aortic enlargement",       # 0
    "Atelectasis",              # 1
    "Calcification",            # 2
    "Cardiomegaly",             # 3
    "Consolidation",            # 4
    "ILD",                      # 5
    "Infiltration",             # 6
    "Lung Opacity",             # 7
    "Nodule/Mass",              # 8
    "Other lesion",             # 9
    "Pleural effusion",         # 10
    "Pleural thickening",       # 11
    "Pneumothorax",             # 12
    "Pulmonary fibrosis",       # 13
    "No finding",               # 14
]
NUM_CLASSES = len(CLASSES)


def _load_dicom_uint8(path: Path) -> np.ndarray:
    """Load DICOM → uint8 grayscale array at ORIGINAL resolution.

    Handles MONOCHROME1 inversion, rescale slope/intercept, and windowing.
    """
    import pydicom

    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)

    slope = getattr(ds, "RescaleSlope", 1)
    intercept = getattr(ds, "RescaleIntercept", 0)
    img = img * slope + intercept

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc, ww = float(ds.WindowCenter), float(ds.WindowWidth)
        lo, hi = wc - ww / 2, wc + ww / 2
    else:
        lo, hi = np.percentile(img, [1, 99])

    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = 1.0 - img

    return (img * 255).astype(np.uint8)


def _preprocess_one(args):
    """Worker: DICOM → resized uint8 .npy + original dims."""
    img_path, cache_path, image_size = args
    img_path, cache_path = Path(img_path), Path(cache_path)
    if cache_path.exists():
        # Already cached — read dims from file shape recorded in dims.csv
        return img_path.stem, None, None
    img = _load_dicom_uint8(img_path)
    orig_h, orig_w = img.shape
    img_resized = np.array(
        Image.fromarray(img, mode="L").resize((image_size, image_size), Image.BILINEAR)
    )
    np.save(cache_path, img_resized)
    return img_path.stem, orig_h, orig_w


class VinDrCXRDataset:
    """VinDr-CXR dataset with multi-label labels and bounding boxes.

    Args:
        root: Dataset root directory (contains train.csv, train/)
        split: "train", "val" or "test"
        image_size: Target image size (square)
        download: Auto-download from Kaggle if missing
        cache_dir: Where to store/load preprocessed .npy images
        val_fraction: Fraction held out as validation
        test_fraction: Fraction held out as test
        seed: Seed for the split (must match across splits)
    """

    def __init__(
        self,
        root: str = "data/vindr_cxr",
        split: str = "train",
        image_size: int = 224,
        download: bool = True,
        cache_dir: str = "data/cache_224",
        val_fraction: float = 0.10,
        test_fraction: float = 0.10,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.classes = CLASSES
        self.num_classes = NUM_CLASSES

        if download and not (self.root / "train.csv").exists():
            download_vindr(str(self.root))

        self.image_ids, self.annotations = self._load_annotations(
            val_fraction, test_fraction, seed
        )
        self.dims = self._load_dims()
        self.train_mode = split == "train"

    def _load_annotations(
        self, val_fraction: float, test_fraction: float, seed: int
    ) -> Tuple[list, Dict[str, Dict]]:
        """Parse train.csv into per-image multi-hot labels + bboxes."""
        csv_path = self.root / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations not found: {csv_path}")

        annotations = defaultdict(lambda: {"labels": np.zeros(NUM_CLASSES, dtype=np.float32),
                                           "bboxes": defaultdict(list)})

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["image_id"]
                class_id = int(row["class_id"])
                class_name = row["class_name"]

                # Multi-label: several rows per image → several 1s in the vector
                annotations[img_id]["labels"][class_id] = 1.0

                if class_name != "No finding":
                    x1 = float(row["x_min"])
                    y1 = float(row["y_min"])
                    x2 = float(row["x_max"])
                    y2 = float(row["y_max"])
                    # Multiple radiologists may annotate the same class → keep all boxes
                    annotations[img_id]["bboxes"][class_name].append([x1, y1, x2, y2])

        # Deterministic 3-way split (same seed → same split everywhere)
        all_ids = sorted(annotations.keys())
        rng = np.random.RandomState(seed)
        shuffled = list(rng.permutation(all_ids))

        n_total = len(all_ids)
        n_test = int(n_total * test_fraction)
        n_val = int(n_total * val_fraction)

        test_ids = set(shuffled[:n_test])
        val_ids = set(shuffled[n_test:n_test + n_val])
        train_ids = set(shuffled[n_test + n_val:])

        if self.split == "train":
            image_ids = [i for i in all_ids if i in train_ids]
        elif self.split == "val":
            image_ids = [i for i in all_ids if i in val_ids]
        else:
            image_ids = [i for i in all_ids if i in test_ids]

        # Convert nested defaultdicts to plain dicts
        clean = {
            img_id: {"labels": ann["labels"], "bboxes": dict(ann["bboxes"])}
            for img_id, ann in annotations.items()
        }
        return image_ids, clean

    def _dims_path(self) -> Path:
        return self.cache_dir / "dims.csv"

    def _load_dims(self) -> Dict[str, Tuple[int, int]]:
        """Load original image dimensions from cache, if available."""
        dims = {}
        path = self._dims_path()
        if path.exists():
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dims[row["image_id"]] = (int(row["orig_h"]), int(row["orig_w"]))
        return dims

    def _append_dims(self, new_dims: Dict[str, Tuple[int, int]]):
        """Append newly computed dims to dims.csv and update in-memory map."""
        if not new_dims:
            return
        write_header = not self._dims_path().exists()
        with open(self._dims_path(), "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["image_id", "orig_h", "orig_w"])
            for img_id, (h, w) in new_dims.items():
                writer.writerow([img_id, h, w])
        self.dims.update(new_dims)

    def build_cache(self, workers: int = 16):
        """Preprocess all images of this split into .npy cache (parallel).

        First run: ~20-40 min for 15k DICOMs. Later runs: instant (skip existing).
        """
        args = [
            (self.root / "train" / f"{img_id}.dicom",
             self.cache_dir / f"{img_id}.npy",
             self.image_size)
            for img_id in self.image_ids
        ]
        missing = [a for a in args if not a[1].exists()]
        if not missing:
            print(f"[cache] All {len(args)} images already cached")
            return

        print(f"[cache] Preprocessing {len(missing)}/{len(args)} images with {workers} workers...")
        new_dims = {}
        with Pool(workers) as pool:
            for img_id, h, w in tqdm(
                pool.imap_unordered(_preprocess_one, missing, chunksize=32),
                total=len(missing), desc="Caching DICOMs",
            ):
                if h is not None:
                    new_dims[img_id] = (h, w)
        self._append_dims(new_dims)
        print(f"[cache] Done. Cache at {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Get one sample.

        Returns:
            image: (H, W, C) float32 — normalized, resized, augmented if train
            labels: (NUM_CLASSES,) float32 multi-hot vector
            bboxes: dict of class_name → [x1, y1, x2, y2] normalized to [0,1]
        """
        img_id = self.image_ids[idx]
        ann = self.annotations[img_id]

        cache_path = self.cache_dir / f"{img_id}.npy"
        if cache_path.exists():
            img = np.load(cache_path)
        else:
            # Slow path: parse DICOM, cache it for next time
            img_path = self.root / "train" / f"{img_id}.dicom"
            img_full = _load_dicom_uint8(img_path)
            self._append_dims({img_id: img_full.shape})
            img = np.array(
                Image.fromarray(img_full, mode="L").resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
            )
            np.save(cache_path, img)

        # Fast on-the-fly: normalize + optional augmentation
        image = augment_normalize(img, train=self.train_mode)

        # Normalize bboxes to [0, 1] using ORIGINAL dimensions
        orig_h, orig_w = self.dims.get(img_id, (self.image_size, self.image_size))
        bboxes_norm = {}
        for class_name, box_list in ann["bboxes"].items():
            bboxes_norm[class_name] = [
                [x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h]
                for (x1, y1, x2, y2) in box_list
            ]

        return image, ann["labels"], bboxes_norm

    def get_class_index(self, class_name: str) -> int:
        """Get index of a class name."""
        return self.classes.index(class_name)
