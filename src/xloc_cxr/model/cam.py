"""Class Activation Mapping methods: Grad-CAM, Grad-CAM++, XGradCAM (Flax NNX)."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def grad_cam(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Standard Grad-CAM.

    Args:
        activations: (H, W, C) feature maps
        gradients: (H, W, C) gradients of class score w.r.t. activations

    Returns:
        (H, W) normalized CAM map in [0, 1]
    """
    weights = gradients.mean(axis=(0, 1))                # (C,)
    cam = np.sum(activations * weights[None, None, :], axis=-1)
    cam = np.maximum(cam, 0)
    return cam / (cam.max() + 1e-8)


def grad_cam_plus_plus(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Grad-CAM++ with higher-order gradient weighting.

    Args:
        activations: (H, W, C) feature maps
        gradients: (H, W, C) gradients

    Returns:
        (H, W) normalized CAM map
    """
    grad_2 = gradients ** 2
    grad_3 = gradients ** 3
    sum_act = activations.sum(axis=(0, 1), keepdims=True)  # (1, 1, C)

    alpha = grad_2 / (2.0 * grad_2 + sum_act * grad_3 + 1e-8)
    weights = np.sum(alpha * np.maximum(gradients, 0), axis=(0, 1))

    cam = np.sum(weights[None, None, :] * activations, axis=-1)
    cam = np.maximum(cam, 0)
    return cam / (cam.max() + 1e-8)


def xgrad_cam(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """XGradCAM with axiom-based normalization.

    Args:
        activations: (H, W, C) feature maps
        gradients: (H, W, C) gradients

    Returns:
        (H, W) normalized CAM map
    """
    sum_act = activations.sum(axis=(0, 1))
    weights = np.sum(gradients * activations, axis=(0, 1)) / (sum_act + 1e-8)

    cam = np.sum(weights[None, None, :] * activations, axis=-1)
    cam = np.maximum(cam, 0)
    return cam / (cam.max() + 1e-8)


CAM_METHODS = {
    "grad_cam": grad_cam,
    "grad_cam_plus_plus": grad_cam_plus_plus,
    "xgrad_cam": xgrad_cam,
}


def _get_tail(model: nnx.Module, layer: str):
    """Return a function mapping layer activations → class scores.

    For layer4: GAP + fc only. For layer3: layer4 blocks + GAP + fc.
    """
    def tail(acts):
        """(1, H, W, C) activations → (num_classes,) scores."""
        if layer == "layer3":
            x = acts
            for block in model.layer4:
                x = block(x)
        else:
            x = acts
        pooled = jnp.mean(x, axis=(1, 2))
        return model.fc(pooled)[0]  # (num_classes,)
    return tail


@partial(nnx.jit, static_argnames=("layer", "target_class"))
def _cam_core(model: nnx.Module, image: jnp.ndarray, target_class: int, layer: str):
    """JIT-compiled: forward + grads of class score w.r.t. chosen layer activations.

    Returns:
        activations: (H, W, C) at chosen layer (batch dim removed)
        gradients: (H, W, C) of score w.r.t. those activations
    """
    logits, layer3_act, layer4_act = model(image)
    acts = layer3_act if layer == "layer3" else layer4_act

    tail = _get_tail(model, layer)
    score_fn = lambda a: tail(a)[target_class]
    _, grads = jax.value_and_grad(score_fn)(acts)

    return acts[0], grads[0]


def get_cam_map(
    model: nnx.Module,
    image: jnp.ndarray,
    target_class: int,
    layer: str = "layer4",
    method: str = "grad_cam",
) -> np.ndarray:
    """Generate CAM map for a specific class and layer.

    Args:
        model: ResNet50 nnx.Module — returns (logits, layer3_act, layer4_act)
        image: (1, H, W, C) input image (channels last)
        target_class: Class index to visualize
        layer: "layer3" or "layer4"
        method: "grad_cam", "grad_cam_plus_plus", or "xgrad_cam"

    Returns:
        (H, W) CAM map at input resolution, in [0, 1]
    """
    if method not in CAM_METHODS:
        raise ValueError(f"Unknown CAM method: {method}. Choose from {list(CAM_METHODS)}")

    acts, grads = _cam_core(model, image, target_class, layer)
    cam = CAM_METHODS[method](np.array(acts), np.array(grads))

    h, w = image.shape[1], image.shape[2]
    cam_upsampled = jax.image.resize(jnp.array(cam), (h, w), "bilinear")
    return np.array(cam_upsampled)
