"""Pointing Game: does the CAM peak fall inside the bbox?"""
import numpy as np


def pointing_game(cam: np.ndarray, bbox: np.ndarray) -> float:
    """Check if the argmax of the CAM falls inside the bounding box.

    Args:
        cam: (H, W) normalized CAM map in [0, 1]
        bbox: [x1, y1, x2, y2] normalized to [0, 1]

    Returns:
        1.0 if peak is inside bbox, 0.0 otherwise
    """
    h, w = cam.shape
    y_peak, x_peak = np.unravel_index(np.argmax(cam), cam.shape)

    # Normalize peak coordinates to [0, 1]
    x_norm = x_peak / w
    y_norm = y_peak / h

    x1, y1, x2, y2 = bbox

    inside = (x1 <= x_norm <= x2) and (y1 <= y_norm <= y2)
    return 1.0 if inside else 0.0
