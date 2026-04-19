"""Tests for :mod:`pheatmap._breaks`."""

from __future__ import annotations

import numpy as np

from pheatmap._breaks import generate_breaks


class TestGenerateBreaks:
    def test_length_n_plus_1(self) -> None:
        br = generate_breaks(np.array([0.0, 1.0]), n=10)
        assert len(br) == 11

    def test_span_matches_input_range(self) -> None:
        x = np.array([-2.5, 3.7, 0.0, 1.2])
        br = generate_breaks(x, n=5)
        assert br[0] == pytest_min(x)
        assert br[-1] == pytest_max(x)

    def test_strictly_monotonic(self) -> None:
        br = generate_breaks(np.array([0.0, 10.0]), n=20)
        assert np.all(np.diff(br) > 0)

    def test_center_true_symmetric(self) -> None:
        x = np.array([-2.0, 5.0])
        br = generate_breaks(x, n=4, center=True)
        assert br[0] == -5.0 and br[-1] == 5.0
        # middle break equals zero for even n
        assert br[2] == 0.0

    def test_ignores_nan(self) -> None:
        x = np.array([np.nan, -1.0, 2.0, np.nan])
        br = generate_breaks(x, n=3)
        assert br[0] == -1.0 and br[-1] == 2.0


def pytest_min(x: np.ndarray) -> float:
    return float(np.nanmin(x))


def pytest_max(x: np.ndarray) -> float:
    return float(np.nanmax(x))
