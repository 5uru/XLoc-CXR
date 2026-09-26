"""Tests for CAM methods."""
import numpy as np
import pytest
import jax.numpy as jnp

from src.xloc_cxr.model.cam import grad_cam, grad_cam_plus_plus, xgrad_cam


class TestGradCAM:
    def test_output_shape(self):
        activations = jnp.ones((7, 7, 64))
        gradients = jnp.ones((7, 7, 64))
        cam = grad_cam(activations, gradients)
        assert cam.shape == (7, 7)

    def test_output_range(self):
        activations = jnp.ones((7, 7, 64))
        gradients = jnp.ones((7, 7, 64))
        cam = grad_cam(activations, gradients)
        assert np.all(cam >= 0)
        assert np.all(cam <= 1)

    def test_empty_input(self):
        activations = jnp.zeros((7, 7, 64))
        gradients = jnp.zeros((7, 7, 64))
        cam = grad_cam(activations, gradients)
        assert np.all(cam == 0)


class TestGradCAMPlusPlus:
    def test_output_shape(self):
        activations = jnp.ones((7, 7, 64))
        gradients = jnp.ones((7, 7, 64))
        cam = grad_cam_plus_plus(activations, gradients)
        assert cam.shape == (7, 7)


class TestXGradCAM:
    def test_output_shape(self):
        activations = jnp.ones((7, 7, 64))
        gradients = jnp.ones((7, 7, 64))
        cam = xgrad_cam(activations, gradients)
        assert cam.shape == (7, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
