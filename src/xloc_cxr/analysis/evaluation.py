"""Run full evaluation: layer × method × class → metrics."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

from ..model.cam import get_cam_map
from ..metrics import (
    pointing_game, energy_in_box, iou_score,
    normalized_distance, spatial_entropy,
)


def run_evaluation(
    model,
    params,
    dataset,
    layers: List[str] = ["layer3", "layer4"],
    cam_methods: List[str] = ["grad_cam", "grad_cam_plus_plus", "xgrad_cam"],
    max_samples: int = 200,
) -> pd.DataFrame:
    """Run full evaluation across layers, methods, and classes.

    Args:
        model: Flax model
        params: Model parameters
        dataset: VinDrCXRDataset
        layers: Which layers to evaluate
        cam_methods: Which CAM methods to evaluate
        max_samples: Max number of samples to evaluate

    Returns:
        DataFrame with columns: [layer, method, class, metric, score]
    """
    results = []
    n_samples = min(max_samples, len(dataset))

    print(f"[evaluation] Running on {n_samples} samples")
    print(f"[evaluation] Layers: {layers}")
    print(f"[evaluation] Methods: {cam_methods}")

    for idx in tqdm(range(n_samples), desc="Evaluating"):
        image, labels, bboxes = dataset[idx]
        image_batch = np.expand_dims(image, axis=0)

        for class_name, has_finding in labels.items():
            if not has_finding:
                continue  # Skip negative classes

            class_idx = dataset.get_class_index(class_name)

            for layer in layers:
                for method in cam_methods:
                    # Generate CAM
                    cam = get_cam_map(
                        model, params, image_batch,
                        target_class=class_idx,
                        layer=layer, method=method,
                    )

                    # Compute metrics
                    if class_name == "No Finding":
                        # No bbox → use diffusivity metrics
                        results.append({
                            "layer": layer, "method": method,
                            "class": class_name, "metric": "entropy",
                            "score": spatial_entropy(cam),
                        })
                        results.append({
                            "layer": layer, "method": method,
                            "class": class_name, "metric": "top_1pct",
                            "score": spatial_entropy(cam),  # placeholder
                        })
                    else:
                        if class_name not in bboxes:
                            continue

                        bbox = bboxes[class_name]

                        results.extend([
                            {"layer": layer, "method": method, "class": class_name,
                             "metric": "pointing_game", "score": pointing_game(cam, bbox)},
                            {"layer": layer, "method": method, "class": class_name,
                             "metric": "energy_box", "score": energy_in_box(cam, bbox)},
                            {"layer": layer, "method": method, "class": class_name,
                             "metric": "iou", "score": iou_score(cam, bbox)},
                            {"layer": layer, "method": method, "class": class_name,
                             "metric": "distance", "score": normalized_distance(cam, bbox)},
                        ])

    df = pd.DataFrame(results)
    print(f"[evaluation] Generated {len(df)} metric scores")
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by layer × method × metric."""
    summary = df.groupby(["layer", "method", "metric"])["score"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    return summary
