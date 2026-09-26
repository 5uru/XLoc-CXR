"""Tests for metric functions."""
import numpy as np
import pytest

from src.xloc_cxr.metrics import (
    pointing_game, energy_in_box, iou_score,
    normalized_distance, spatial_entropy,
)


class TestPointingGame:
    def test_peak_inside(self):
        cam = np.zeros((100, 100))
        cam[50, 50] = 1.0  # Peak at center
        bbox = [0.4, 0.4, 0.6, 0.6]  # Contains center
        assert pointing_game(cam, bbox) == 1.0

    def test_peak_outside(self):
        cam = np.zeros((100, 100))
        cam[10, 10] = 1.0  # Peak at corner
        bbox = [0.4, 0.4, 0.6, 0.6]  # Does not contain corner
        assert pointing_game(cam, bbox) == 0.0


class TestEnergyBox:
    def test_all_energy_inside(self):
        cam = np.zeros((100, 100))
        cam[45:55, 45:55] = 1.0  # All energy in bbox region
        bbox = [0.4, 0.4, 0.6, 0.6]
        score = energy_in_box(cam, bbox)
        assert score > 0.5  # Most energy should be in box

    def test_no_energy_inside(self):
        cam = np.zeros((100, 100))
        cam[10:20, 10:20] = 1.0  # Energy outside bbox
        bbox = [0.4, 0.4, 0.6, 0.6]
        score = energy_in_box(cam, bbox)
        assert score == 0.0


class TestIoU:
    def test_perfect_overlap(self):
        cam = np.zeros((100, 100))
        cam[40:60, 40:60] = 1.0  # Perfectly matches bbox
        bbox = [0.4, 0.4, 0.6, 0.6]
        score = iou_score(cam, bbox, threshold=0.5)
        assert score == 1.0

    def test_no_overlap(self):
        cam = np.zeros((100, 100))
        cam[10:20, 10:20] = 1.0
        bbox = [0.4, 0.4, 0.6, 0.6]
        score = iou_score(cam, bbox, threshold=0.5)
        assert score == 0.0


class TestDistance:
    def test_zero_distance(self):
        cam = np.zeros((100, 100))
        cam[50, 50] = 1.0  # Peak at bbox center
        bbox = [0.4, 0.4, 0.6, 0.6]  # Center at (0.5, 0.5)
        dist = normalized_distance(cam, bbox)
        assert dist < 0.1


class TestDiffusivity:
    def test_uniform_high_entropy(self):
        cam = np.ones((100, 100))  # Uniform = high entropy
        entropy = spatial_entropy(cam)
        assert entropy > 10  # Should be high

    def test_peaked_low_entropy(self):
        cam = np.zeros((100, 100))
        cam[50, 50] = 1.0  # Peaked = low entropy
        entropy = spatial_entropy(cam)
        assert entropy < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
