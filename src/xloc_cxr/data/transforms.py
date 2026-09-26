"""Image transformations for CXR preprocessing."""
import numpy as np

# ImageNet stats
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def augment_normalize(img: np.ndarray, train: bool = False) -> np.ndarray:
    """Normalize a uint8 grayscale image to (H, W, 3) float32 for the model.

    Args:
        img: (H, W) uint8 grayscale image
        train: Apply horizontal flip augmentation

    Returns:
        (H, W, 3) float32 normalized array (channels-last, NNX convention)
    """
    img = img.astype(np.float32) / 255.0

    # Grayscale → 3 channels
    img = np.stack([img] * 3, axis=-1)

    # ImageNet normalization
    img = (img - MEAN) / STD

    # Augmentation: horizontal flip (train only)
    if train and np.random.random() > 0.5:
        img = img[:, ::-1, :]

    return img
