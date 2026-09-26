"""Tests for VinDr-CXR dataset loading and splitting."""
import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.xloc_cxr.data.vindr import VinDrCXRDataset, CLASSES, NUM_CLASSES


def _write_fake_csv(root: Path, n_images: int = 30):
    """Create a minimal train.csv with fake annotations."""
    rows = []
    for i in range(n_images):
        img_id = f"img_{i:04d}"
        if i % 3 == 0:
            # "No finding" row
            rows.append({
                "image_id": img_id, "class_name": "No finding",
                "class_id": 14, "rad_id": "R1",
                "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0,
            })
        else:
            # A finding with a bbox
            cls_id = i % 14
            rows.append({
                "image_id": img_id, "class_name": CLASSES[cls_id],
                "class_id": cls_id, "rad_id": "R1",
                "x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200,
            })

    with open(root / "train.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestSplitConsistency:
    """Train / val / test splits must be disjoint and cover all images."""

    def test_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            _write_fake_csv(root, n_images=30)

            train_ds = VinDrCXRDataset(root=str(root), split="train", download=False)
            val_ds = VinDrCXRDataset(root=str(root), split="val", download=False)
            test_ds = VinDrCXRDataset(root=str(root), split="test", download=False)

            train_ids = set(train_ds.image_ids)
            val_ids = set(val_ds.image_ids)
            test_ids = set(test_ds.image_ids)

            # All disjoint
            assert train_ids.isdisjoint(val_ids), "train/val overlap!"
            assert train_ids.isdisjoint(test_ids), "train/test overlap!"
            assert val_ids.isdisjoint(test_ids), "val/test overlap!"

            # Complete coverage
            assert len(train_ids) + len(val_ids) + len(test_ids) == 30

    def test_split_sizes(self):
        """Default: 80% train, 10% val, 10% test."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            _write_fake_csv(root, n_images=100)

            train_ds = VinDrCXRDataset(root=str(root), split="train", download=False)
            val_ds = VinDrCXRDataset(root=str(root), split="val", download=False)
            test_ds = VinDrCXRDataset(root=str(root), split="test", download=False)

            assert len(test_ds) == 10
            assert len(val_ds) == 10
            assert len(train_ds) == 80

    def test_deterministic(self):
        """Same seed must give the same split."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            _write_fake_csv(root, n_images=30)

            ds1 = VinDrCXRDataset(root=str(root), split="test", download=False)
            ds2 = VinDrCXRDataset(root=str(root), split="test", download=False)
            assert ds1.image_ids == ds2.image_ids


class TestCache:
    def test_getitem_uses_cache(self):
        """getitem must return (H, W, 3) float32 and cache the npy file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            cache = Path(tmp) / "cache"
            (root / "train").mkdir(parents=True)
            _write_fake_csv(root, n_images=5)

            # Create fake cached images (uint8 224x224)
            cache.mkdir()
            for i in range(5):
                np.save(cache / f"img_{i:04d}.npy", np.zeros((224, 224), dtype=np.uint8))

            ds = VinDrCXRDataset(
                root=str(root), split="train", download=False,
                cache_dir=str(cache),
            )
            img, labels, bboxes = ds[0]
            assert img.shape == (224, 224, 3)
            assert img.dtype == np.float32
            assert labels.shape == (NUM_CLASSES,)
            assert isinstance(bboxes, dict)


class TestAnnotationParsing:
    def test_label_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            _write_fake_csv(root, n_images=10)

            ds = VinDrCXRDataset(root=str(root), split="train", download=False)
            ann = ds.annotations["img_0000"]  # "No finding"
            assert ann["labels"][14] == 1.0
            assert ann["labels"].sum() == 1.0
            assert len(ann["bboxes"]) == 0  # No finding → no bbox

            ann2 = ds.annotations["img_0001"]  # Atelectasis (id 1)
            assert ann2["labels"][1] == 1.0
            assert "Atelectasis" in ann2["bboxes"]
            # bboxes are now lists (one per radiologist annotation)
            assert ann2["bboxes"]["Atelectasis"] == [[10.0, 20.0, 100.0, 200.0]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
