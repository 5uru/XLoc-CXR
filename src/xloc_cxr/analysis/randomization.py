"""Weight randomization test + controls, PER PATHOLOGY (Flax NNX).

Controls implemented (per the study protocol):
  1. TRAINED model          — real scores
  2. UNTRAINED model        — freshly initialized network, same architecture
  3. RANDOMIZED weights     — trained weights randomly permuted (destruction test)
  4. RANDOM CAM             — uniform noise maps scored against bboxes (floor level)

All comparisons are reported globally AND per class, since localization
performance varies strongly across pathologies (large vs small lesions).
"""
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
import jax.numpy as jnp
from flax import nnx
from scipy import stats

from ..model.cam import get_cam_map
from ..metrics import pointing_game, energy_in_box


def randomize_weights(model: nnx.Module, seed: int = 42):
    """Randomly permute all weight parameters in-place (keeps biases).

    Destroys learned features while preserving the per-layer weight
    distribution — the classic sanity check for saliency methods.
    """
    rng = np.random.RandomState(seed)
    state = nnx.to_pure_dict(nnx.state(model, nnx.Param))

    def permute(tree):
        if isinstance(tree, dict):
            return {k: permute(v) for k, v in tree.items()}
        arr = np.asarray(tree)
        if arr.ndim >= 2:
            flat = arr.reshape(arr.shape[0], -1)
            perm = rng.permutation(flat.shape[0])
            return jnp.array(flat[perm].reshape(arr.shape))
        return tree

    nnx.update(model, permute(state))


def _scores_for_model(model: nnx.Module, dataset, cam_method: str,
                      layer: str, n_samples: int) -> Dict[str, List[float]]:
    """Per-class mean pointing-game + energy-box (best across bboxes).

    Returns:
        dict: class_name → list of per-(image, class) scores
    """
    model.eval()
    per_class = defaultdict(list)

    for i in range(min(n_samples, len(dataset))):
        image, labels_vec, bboxes = dataset[i]
        image_batch = jnp.array(np.expand_dims(image, axis=0))

        for class_idx, has_finding in enumerate(labels_vec):
            class_name = dataset.classes[class_idx]
            if not has_finding or class_name not in bboxes:
                continue

            cam = get_cam_map(model, image_batch, class_idx,
                              layer=layer, method=cam_method)
            box_list = bboxes[class_name]

            pg = max(pointing_game(cam, np.array(b)) for b in box_list)
            eb = max(energy_in_box(cam, np.array(b)) for b in box_list)
            per_class[class_name].append((pg + eb) / 2.0)

    return dict(per_class)


def _scores_random_cam(dataset, n_samples: int, seed: int = 0) -> Dict[str, List[float]]:
    """Floor control: uniform random CAM maps scored against the same bboxes."""
    rng = np.random.RandomState(seed)
    per_class = defaultdict(list)

    for i in range(min(n_samples, len(dataset))):
        _, labels_vec, bboxes = dataset[i]
        for class_idx, has_finding in enumerate(labels_vec):
            class_name = dataset.classes[class_idx]
            if not has_finding or class_name not in bboxes:
                continue
            cam = rng.rand(224, 224).astype(np.float32)
            box_list = bboxes[class_name]
            pg = max(pointing_game(cam, np.array(b)) for b in box_list)
            eb = max(energy_in_box(cam, np.array(b)) for b in box_list)
            per_class[class_name].append((pg + eb) / 2.0)

    return dict(per_class)


def _compare(real: List[float], null: List[float]) -> Dict:
    """Mann-Whitney U test (one-sided: real > null)."""
    if len(real) < 3 or len(null) < 3:
        return {"n": len(real), "real_mean": float(np.mean(real)) if real else float("nan"),
                "null_mean": float(np.mean(null)) if null else float("nan"),
                "p_value": float("nan"), "significant": False}
    u, p = stats.mannwhitneyu(real, null, alternative="greater")
    return {
        "n": len(real),
        "real_mean": float(np.mean(real)),
        "real_std": float(np.std(real)),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def run_controls(
    trained_model: nnx.Module,
    fresh_model_fn: Callable,
    dataset,
    cam_method: str = "grad_cam",
    layer: str = "layer4",
    n_samples: int = 100,
    seed: int = 42,
) -> Dict:
    """Run the full control suite, globally and per pathology.

    Args:
        trained_model: model with trained weights loaded
        fresh_model_fn: callable returning a new untrained model
        dataset: evaluation dataset (test split)
        cam_method: CAM method to test
        layer: layer to test
        n_samples: number of test images per condition
        seed: random seed

    Returns:
        {
          "global": {condition → stats},
          "per_class": {class_name → {condition → stats}},
        }
    """
    conditions = {}

    print(f"\n[controls] 1/4 TRAINED model ({n_samples} samples)...")
    conditions["trained"] = _scores_for_model(
        trained_model, dataset, cam_method, layer, n_samples)

    print("[controls] 2/4 UNTRAINED model...")
    untrained = fresh_model_fn()
    conditions["untrained"] = _scores_for_model(
        untrained, dataset, cam_method, layer, n_samples)

    print("[controls] 3/4 RANDOMIZED weights...")
    randomize_weights(untrained, seed=seed)
    conditions["randomized_weights"] = _scores_for_model(
        untrained, dataset, cam_method, layer, n_samples)

    print("[controls] 4/4 RANDOM CAM maps...")
    conditions["random_cam"] = _scores_random_cam(dataset, n_samples, seed=seed)

    # --- Global comparison (all classes pooled) ---
    real_all = [s for scores in conditions["trained"].values() for s in scores]
    global_results = {
        "trained": {"mean": float(np.mean(real_all)),
                    "std": float(np.std(real_all)), "n": len(real_all)},
    }
    for name in ("untrained", "randomized_weights", "random_cam"):
        null_all = [s for scores in conditions[name].values() for s in scores]
        global_results[name] = _compare(real_all, null_all)

    # --- Per-class comparison ---
    per_class_results = {}
    for class_name, real_scores in conditions["trained"].items():
        entry = {"trained": {"mean": float(np.mean(real_scores)),
                             "std": float(np.std(real_scores)),
                             "n": len(real_scores)}}
        for name in ("untrained", "randomized_weights", "random_cam"):
            null_scores = conditions[name].get(class_name, [])
            entry[name] = _compare(real_scores, null_scores)
        per_class_results[class_name] = entry

    return {"global": global_results, "per_class": per_class_results}
