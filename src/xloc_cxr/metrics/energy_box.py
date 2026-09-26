"""Energy in Box: fraction of CAM activation inside the bounding box."""
import numpy as np


def energy_in_box(cam: np.ndarray, bbox: np.ndarray) -> float:
    """Compute the fraction of total CAM energy inside the bbox.

    Args:
        cam: (H, W) normalized CAM map
        bbox: [x1, y1, x2, y2] normalized to [0, 1]

    Returns:
        Fraction in [0, 1]
    """
    h, w = cam.shape

    # Convert normalized bbox to pixel coordinates
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)

    # Clip to image bounds
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    total_energy = np.sum(cam)
    if total_energy < 1e-8:
        return 0.0

    box_energy = np.sum(cam[y1:y2, x1:x2])
    return float(box_energy / total_energy)
