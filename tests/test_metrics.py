"""Numerical-correctness tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from implicitlidar.eval import (
    box_detections_from_hits,
    chamfer_distance,
    frechet_distance,
    miss_rate,
)


def test_chamfer_distance_zero_for_identical_clouds():
    pts = np.random.RandomState(0).randn(50, 3)
    assert chamfer_distance(pts, pts) == pytest.approx(0.0, abs=1e-10)


def test_chamfer_distance_symmetric():
    a = np.random.RandomState(0).randn(50, 3)
    b = np.random.RandomState(1).randn(50, 3)
    assert chamfer_distance(a, b) == pytest.approx(chamfer_distance(b, a))


def test_frechet_distance_zero_for_identical_curves():
    p = np.linspace([0, 0, 0], [1, 1, 1], 20)
    assert frechet_distance(p, p) == pytest.approx(0.0, abs=1e-10)


def test_frechet_distance_translated_curves():
    p = np.linspace([0, 0, 0], [1, 0, 0], 20)
    q = p + np.array([0, 0.5, 0])
    assert frechet_distance(p, q) == pytest.approx(0.5, abs=1e-6)


def test_miss_rate_basic():
    detected = np.array([True, True, False, False, True])
    assert miss_rate(detected, n_ground_truth=5) == pytest.approx(2 / 5)


def test_box_detections_inside_outside():
    boxes = [(np.array([0, 0, 0]), np.array([1, 1, 1])),
             (np.array([2, 2, 2]), np.array([3, 3, 3]))]
    hits = np.array([[0.5, 0.5, 0.5], [10, 10, 10]])
    detected = box_detections_from_hits(hits, boxes)
    assert detected.tolist() == [True, False]
