"""Evaluation script for CAM metrics (Flax NNX)."""
import numpy as np
import jax.numpy as jnp
import pandas as pd
from flax import nnx
from tqdm import tqdm

from .data import VinDrCXRDataset
from .model import create_model
from .model.cam import get_cam_map
from .metrics import (
    pointing_game, energy_in_box, iou_score,
    normalized_distance, spatial_entropy, top_k_concentration,
)


def _score_bbox_metrics(cam: np.ndarray, bboxes: list) -> dict:
    """Score a CAM against one or more ground-truth bboxes.

    With multiple radiologist boxes for the same class, we take the BEST
    score across boxes (standard practice: the CAM is correct if it matches
    at least one annotated region).
    """
    def best(metric_fn):
        return max(metric_fn(cam, np.array(b)) for b in bboxes)

    return {
        "pointing_game": best(pointing_game),
        "energy_box": best(energy_in_box),
        "iou": best(iou_score),
        "distance": min(normalized_distance(cam, np.array(b)) for b in bboxes),
    }


def evaluate(config: dict, checkpoint: str, max_samples: int = 500):
    """Run CAM evaluation on the held-out test set."""
    print("=" * 60)
    print("  XLoc-CXR Evaluation (Flax NNX)")
    print("=" * 60)

    # Load model
    print("\n[evaluate] Creating model...")
    model, _ = create_model(num_classes=config["data"]["num_classes"])

    if checkpoint:
        print(f"[evaluate] Loading checkpoint: {checkpoint}")
        import pickle
        with open(checkpoint, "rb") as f:
            nnx.update(model, pickle.load(f))

    model.eval()

    # Load test set
    print("\n[evaluate] Loading test dataset...")
    dataset = VinDrCXRDataset(
        root=config["data"]["root"],
        split="test",
        image_size=config["data"]["image_size"],
    )
    dataset.build_cache()
    print(f"[evaluate] Test samples: {len(dataset)}")

    layers = config["eval"]["layers"]
    methods = config["eval"]["cam_methods"]
    max_samples = min(max_samples, len(dataset))

    results = []
    print(f"\n[evaluate] Grid: {len(layers)} layers × {len(methods)} methods on {max_samples} samples")

    for idx in tqdm(range(max_samples), desc="Evaluating"):
        image, labels_vec, bboxes = dataset[idx]
        image_batch = jnp.array(np.expand_dims(image, axis=0))

        for class_idx, has_finding in enumerate(labels_vec):
            if not has_finding:
                continue
            class_name = dataset.classes[class_idx]

            for layer in layers:
                for method in methods:
                    cam = get_cam_map(
                        model, image_batch,
                        target_class=class_idx,
                        layer=layer, method=method,
                    )

                    if class_name == "No finding":
                        # No bbox → diffusivity metrics only
                        results.append({
                            "image_idx": idx, "layer": layer, "method": method,
                            "class": class_name, "metric": "entropy",
                            "score": spatial_entropy(cam),
                        })
                        results.append({
                            "image_idx": idx, "layer": layer, "method": method,
                            "class": class_name, "metric": "top_1pct",
                            "score": top_k_concentration(cam, k=0.01),
                        })
                    elif class_name in bboxes:
                        scores = _score_bbox_metrics(cam, bboxes[class_name])
                        for metric_name, value in scores.items():
                            results.append({
                                "image_idx": idx, "layer": layer, "method": method,
                                "class": class_name, "metric": metric_name,
                                "score": value,
                            })

    df = pd.DataFrame(results)
    df.to_csv("outputs/results.csv", index=False)
    print(f"\n[evaluate] Saved {len(df)} rows to outputs/results.csv")

    summary = df.groupby(["layer", "method", "metric"])["score"].agg(["mean", "std", "count"])
    print("\n[evaluate] Summary:")
    print(summary.to_string())

    return df
