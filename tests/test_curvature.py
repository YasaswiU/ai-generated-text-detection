"""Tests for backend/app/features/curvature.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.features.curvature import compute_curvature_features


def test_flat_sequence_has_zero_curvature():
    scores = [10.0, 10.0, 10.0, 10.0, 10.0]
    features = compute_curvature_features(scores)
    assert features["mean_first_difference"] == 0.0
    assert features["max_absolute_curvature"] == 0.0


def test_monotonic_sequence():
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    features = compute_curvature_features(scores)
    assert features["mean_first_difference"] == 10.0
    # Second differences are ~0 for a perfectly linear sequence.
    assert abs(features["mean_second_difference"]) < 1e-9


def test_too_few_segments_returns_zeros():
    features = compute_curvature_features([5.0])
    assert features["mean_first_difference"] == 0.0
    assert features["curvature_sign_changes"] == 0


def test_oscillating_sequence_has_sign_changes():
    scores = [10.0, 30.0, 10.0, 30.0, 10.0]
    features = compute_curvature_features(scores)
    assert features["max_absolute_curvature"] > 0
    assert features["curvature_sign_changes"] >= 0
