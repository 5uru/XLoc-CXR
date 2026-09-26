"""ResNet50 model for multi-label CXR classification using Flax NNX."""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import Tuple


class BasicBlock(nnx.Module):
    """Basic ResNet block (2 conv layers with skip connection)."""

    def __init__(self, in_features: int, out_features: int, stride: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features, out_features,
            kernel_size=(3, 3), strides=(stride, stride),
            padding="SAME", rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(out_features, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_features, out_features,
            kernel_size=(3, 3), strides=(1, 1),
            padding="SAME", rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(out_features, rngs=rngs)

        # Skip connection projection if dimensions change
        self.needs_projection = stride != 1 or in_features != out_features
        if self.needs_projection:
            self.conv_proj = nnx.Conv(
                in_features, out_features,
                kernel_size=(1, 1), strides=(stride, stride),
                padding="SAME", rngs=rngs,
            )
            self.bn_proj = nnx.BatchNorm(out_features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.needs_projection:
            residual = self.conv_proj(residual)
            residual = self.bn_proj(residual)

        return nnx.relu(x + residual)


class ResNet50(nnx.Module):
    """ResNet50 with multi-label classification head.

    Input format: (B, H, W, C) — NNX convention (channels last).

    Returns logits plus intermediate activations from layer3 and layer4
    for CAM computation.
    """

    def __init__(self, num_classes: int = 14, *, rngs: nnx.Rngs):
        self.stem_conv = nnx.Conv(
            3, 64, kernel_size=(7, 7), strides=(2, 2),
            padding="SAME", rngs=rngs,
        )
        self.stem_bn = nnx.BatchNorm(64, rngs=rngs)

        # layer1: 3 blocks, 64 -> 64, no downsample
        self.layer1 = nnx.List([BasicBlock(64, 64, 1, rngs=rngs) for _ in range(3)])
        # layer2: 4 blocks, 64 -> 128, downsample first
        self.layer2 = nnx.List([BasicBlock(64 if i == 0 else 128, 128, 2 if i == 0 else 1, rngs=rngs) for i in range(4)])
        # layer3: 6 blocks, 128 -> 256, downsample first  (CAM layer 3)
        self.layer3 = nnx.List([BasicBlock(128 if i == 0 else 256, 256, 2 if i == 0 else 1, rngs=rngs) for i in range(6)])
        # layer4: 3 blocks, 256 -> 512, downsample first  (CAM layer 4)
        self.layer4 = nnx.List([BasicBlock(256 if i == 0 else 512, 512, 2 if i == 0 else 1, rngs=rngs) for i in range(3)])

        self.fc = nnx.Linear(512, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: (B, H, W, C) input

        Returns:
            logits: (B, num_classes)
            layer3_out: (B, H3, W3, 256)
            layer4_out: (B, H4, W4, 512)
        """
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)

        layer3_out = x
        for block in self.layer3:
            layer3_out = block(layer3_out)

        layer4_out = layer3_out
        for block in self.layer4:
            layer4_out = block(layer4_out)

        # Global average pooling + classifier
        pooled = jnp.mean(layer4_out, axis=(1, 2))
        logits = self.fc(pooled)

        return logits, layer3_out, layer4_out


def create_model(num_classes: int = 14, seed: int = 42) -> Tuple[nnx.Module, nnx.Optimizer]:
    """Create model and optimizer.

    Returns:
        model: ResNet50 nnx.Module
        optimizer: nnx.Optimizer wrapping the model
    """
    rngs = nnx.Rngs(seed)
    model = ResNet50(num_classes=num_classes, rngs=rngs)
    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=1e-3, weight_decay=1e-4), wrt=nnx.Param
    )
    return model, optimizer


def binary_cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss for multi-label classification."""
    if logits.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}. "
            f"Check that config num_classes == dataset num_classes."
        )
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute per-sample exact-match accuracy."""
    predictions = jax.nn.sigmoid(logits) > 0.5
    return jnp.mean((predictions == (labels > 0.5)).all(axis=1))
