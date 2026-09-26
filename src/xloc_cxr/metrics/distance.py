"""Normalized distance from CAM peak to bbox center."""
import numpy as np


def normalized_distance(cam: np.ndarray, bbox: np.ndarray) -> float:
    """Compute distance from CAM argmax to bbox center, normalized by bbox diagonal.

    Args:
        cam: (H, W) normalized CAM map
        bbox: [x1, y1, x2, y2] normalized to [0, 1]

    Returns:
        Normalized distance (lower is better)
    """
    h, w = cam.shape
    y_peak, x_peak = np.unravel_index(np.argmax(cam), cam.shape)

    # Normalize peak to [0, 1]
    x_norm = x_peak / w
    y_norm = y_peak / h

    # Bbox center
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    # Bbox diagonal (in normalized coords)
    diag = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)

    if diag < 1e-8:
        return 1.0

    dist = np.sqrt((x_norm - cx) ** 2 + (y_norm - cy) ** 2)
    return float(dist / diag)
