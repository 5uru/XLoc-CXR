"""Contrast successful vs failed CAM localizations (key paper figure)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import jax.numpy as jnp

from ..model.cam import get_cam_map
from ..metrics import pointing_game
from .cam_examples import _load_original_image, _draw_bboxes


def _score_image_class(model, dataset, idx, class_idx, layer, method):
    """Return (cam, bboxes, pg_score) for one (image, class)."""
    image, _, bboxes = dataset[idx]
    class_name = dataset.classes[class_idx]
    cam = get_cam_map(
        model, jnp.array(np.expand_dims(image, axis=0)),
        target_class=class_idx, layer=layer, method=method,
    )
    box_list = bboxes.get(class_name, [])
    if not box_list:
        return cam, [], 0.0
    pg = max(pointing_game(cam, np.array(b)) for b in box_list)
    return cam, box_list, pg


def find_examples(
    model, dataset,
    success_class="Cardiomegaly",
    failure_class="Nodule/Mass",
    layer="layer4", method="grad_cam_plus_plus",
    n_each=3,
):
    """Find n success examples (PG=1) and n failure examples (PG=0)."""
    successes, failures = [], []

    for idx in range(len(dataset)):
        _, labels_vec, bboxes = dataset[idx]
        for class_idx, present in enumerate(labels_vec):
            cname = dataset.classes[class_idx]
            if not present or cname not in bboxes:
                continue

            if cname == success_class and len(successes) < n_each:
                cam, boxes, pg = _score_image_class(model, dataset, idx, class_idx, layer, method)
                if pg == 1.0:
                    successes.append((idx, class_idx, cname, cam, boxes, pg))

            elif cname == failure_class and len(failures) < n_each:
                cam, boxes, pg = _score_image_class(model, dataset, idx, class_idx, layer, method)
                if pg == 0.0:
                    failures.append((idx, class_idx, cname, cam, boxes, pg))

        if len(successes) >= n_each and len(failures) >= n_each:
            break

    return successes, failures


def plot_success_failure(
    model, dataset,
    success_class="Cardiomegaly",
    failure_class="Nodule/Mass",
    layer="layer4", method="grad_cam_plus_plus",
    n_each=3,
    output_dir="outputs/figures",
):
    """Figure: rows = success examples (top) vs failure examples (bottom).

    Each row: [original | CAM overlay | CAM + GT bbox + score]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[success_failure] Finding {n_each} successes ({success_class}) + {n_each} failures ({failure_class})...")
    successes, failures = find_examples(
        model, dataset, success_class, failure_class, layer, method, n_each,
    )

    if not successes and not failures:
        print("[success_failure] No examples found - check class names")
        return

    rows = []
    for ex in successes:
        rows.append(("SUCCESS", ex))
    for ex in failures:
        rows.append(("FAILURE", ex))

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)

    for row, (status, (idx, class_idx, class_name, cam, boxes, pg)) in enumerate(rows):
        image_id = dataset.image_ids[idx]
        orig = _load_original_image(dataset, image_id)

        # Original
        axes[row, 0].imshow(orig, cmap="gray")
        axes[row, 0].set_title(f"{class_name}\n(original)", fontsize=11)
        axes[row, 0].axis("off")

        # CAM overlay
        axes[row, 1].imshow(orig, cmap="gray")
        axes[row, 1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row, 1].set_title(f"{method} ({layer})", fontsize=11)
        axes[row, 1].axis("off")

        # CAM + bbox + score, color-coded by success
        axes[row, 2].imshow(orig, cmap="gray")
        axes[row, 2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        _draw_bboxes(axes[row, 2], boxes, size=orig.shape[0], color="lime")
        color = "green" if status == "SUCCESS" else "red"
        axes[row, 2].set_title(
            f"{status}: Pointing Game = {pg:.0f}", fontsize=11, color=color, fontweight="bold",
        )
        axes[row, 2].axis("off")

    plt.suptitle(
        f"Successful vs failed localizations  -  success={success_class}, failure={failure_class}",
        fontsize=13, y=1.002,
    )
    plt.tight_layout()
    out = f"{output_dir}/success_vs_failure.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[success_failure] Saved {out} ({len(successes)} successes, {len(failures)} failures)")
