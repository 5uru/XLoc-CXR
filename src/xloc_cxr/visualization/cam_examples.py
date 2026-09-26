"""Visualize CAM overlays with ground-truth bboxes (publication figures)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from ..data.vindr import _load_dicom_uint8


def _load_original_image(dataset, image_id: str, size: int = 224) -> np.ndarray:
    """Load the original DICOM as a displayable RGB array (uint8)."""
    dcm_path = dataset.root / "train" / f"{image_id}.dicom"
    img = _load_dicom_uint8(dcm_path)
    img = np.array(Image.fromarray(img, mode="L").resize((size, size), Image.BILINEAR))
    return np.stack([img] * 3, axis=-1)  # grayscale → RGB


def _draw_bboxes(ax, bboxes: list, size: int, color: str):
    """Draw all ground-truth boxes for one class."""
    for (x1, y1, x2, y2) in bboxes:
        rect = mpatches.Rectangle(
            (x1 * size, y1 * size),
            (x2 - x1) * size, (y2 - y1) * size,
            fill=False, edgecolor=color, linewidth=2,
        )
        ax.add_patch(rect)


def plot_cam_grid(
    dataset,
    model,
    n_examples: int = 6,
    layer: str = "layer4",
    method: str = "grad_cam_plus_plus",
    output_dir: str = "outputs/figures",
    seed: int = 0,
):
    """Grid of examples: original image | CAM overlay | CAM with bbox.

    Selects test images that have at least one finding with a bbox.
    """
    import jax.numpy as jnp
    from ..model.cam import get_cam_map

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find samples with bbox findings
    rng = np.random.RandomState(seed)
    candidates = []
    for i in range(len(dataset)):
        _, labels_vec, bboxes = dataset[i]
        for class_idx, present in enumerate(labels_vec):
            cname = dataset.classes[class_idx]
            if present and cname in bboxes:
                candidates.append((i, class_idx, cname))
                break
    rng.shuffle(candidates)
    candidates = candidates[:n_examples]

    if not candidates:
        print("[cam_examples] No samples with bboxes found")
        return

    n = len(candidates)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)

    for row, (idx, class_idx, class_name) in enumerate(candidates):
        image, _, bboxes = dataset[idx]
        image_id = dataset.image_ids[idx]

        # Original DICOM for display (not the normalized tensor)
        orig = _load_original_image(dataset, image_id)

        # CAM
        cam = get_cam_map(
            model, jnp.array(np.expand_dims(image, axis=0)),
            target_class=class_idx, layer=layer, method=method,
        )

        # Col 1: original
        axes[row, 0].imshow(orig, cmap="gray")
        axes[row, 0].set_title(f"{class_name}\n(original)")
        axes[row, 0].axis("off")

        # Col 2: CAM overlay
        axes[row, 1].imshow(orig, cmap="gray")
        axes[row, 1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row, 1].set_title(f"{method}\n({layer})")
        axes[row, 1].axis("off")

        # Col 3: CAM + ground-truth bboxes
        axes[row, 2].imshow(orig, cmap="gray")
        axes[row, 2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        _draw_bboxes(axes[row, 2], bboxes[class_name], size=orig.shape[0], color="lime")
        axes[row, 2].set_title("Ground truth bbox")
        axes[row, 2].axis("off")

    plt.tight_layout()
    out = f"{output_dir}/cam_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[cam_examples] Saved {out}")
