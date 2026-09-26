"""Spatial diffusivity metrics for No Finding class (no bbox available)."""
import numpy as np
from scipy import stats


def spatial_entropy(cam: np.ndarray, bins: int = 50) -> float:
    """Compute spatial entropy of CAM map.

    Higher entropy = more diffuse attention.
    Lower entropy = more concentrated attention.

    Args:
        cam: (H, W) normalized CAM map
        bins: Number of histogram bins

    Returns:
        Entropy value
    """
    # Flatten and normalize to probability distribution
    cam_flat = cam.flatten()
    cam_flat = cam_flat / (np.sum(cam_flat) + 1e-8)

    # Remove zeros for log computation
    cam_flat = cam_flat[cam_flat > 1e-10]

    if len(cam_flat) == 0:
        return 0.0

    entropy = -np.sum(cam_flat * np.log2(cam_flat))
    return float(entropy)


def top_k_concentration(cam: np.ndarray, k: float = 0.01) -> float:
    """Fraction of energy in top k% of pixels.

    Args:
        cam: (H, W) CAM map
        k: Fraction of top pixels to consider

    Returns:
        Energy fraction in [0, 1]
    """
    cam_flat = cam.flatten()
    n_top = max(1, int(len(cam_flat) * k))
    top_values = np.sort(cam_flat)[-n_top:]
    total = np.sum(cam_flat) + 1e-8
    return float(np.sum(top_values) / total)
