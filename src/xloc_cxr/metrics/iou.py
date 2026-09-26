"""IoU between binarized CAM and bounding box."""
import numpy as np


def iou_score(cam: np.ndarray, bbox: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU between thresholded CAM and ground truth bbox.

    Args:
        cam: (H, W) normalized CAM map
        bbox: [x1, y1, x2, y2] normalized to [0, 1]
        threshold: Binarization threshold for CAM

    Returns:
        IoU in [0, 1]
    """
    h, w = cam.shape

    # Binarize CAM
    cam_binary = (cam >= threshold).astype(np.float32)

    # Create bbox mask
    bbox_mask = np.zeros((h, w), dtype=np.float32)
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)

    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask[y1:y2, x1:x2] = 1.0

    # Compute IoU
    intersection = np.sum(cam_binary * bbox_mask)
    union = np.sum(cam_binary) + np.sum(bbox_mask) - intersection

    if union < 1e-8:
        return 0.0

    return float(intersection / union)
