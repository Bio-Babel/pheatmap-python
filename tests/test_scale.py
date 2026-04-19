"""Tests for :mod:`pheatmap._scale`."""

from __future__ import annotations

import numpy as np
import pytest

from pheatmap._scale import scale_mat, scale_rows


class TestScaleRows:
    def test_zero_mean_unit_sd(self) -> None:
        x = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])
        out = scale_rows(x)
        np.testing.assert_allclose(np.nanmean(out, axis=1), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.nanstd(out, axis=1, ddof=1), [1.0, 1.0], atol=1e-12)

    def test_nan_ignored(self) -> None:
        x = np.array([[1.0, np.nan, 3.0]])
        out = scale_rows(x)
        assert np.isnan(out[0, 1])
        assert np.isfinite(out[0, 0]) and np.isfinite(out[0, 2])


class TestScaleMat:
    def test_none_returns_copy_as_float(self) -> None:
        x = np.array([[1, 2], [3, 4]])
        out = scale_mat(x, "none")
        assert out.dtype == float
        np.testing.assert_array_equal(out, x.astype(float))

    def test_row_matches_scale_rows(self) -> None:
        x = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])
        np.testing.assert_allclose(scale_mat(x, "row"), scale_rows(x))

    def test_column_scales_columns(self) -> None:
        x = np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]])
        out = scale_mat(x, "column")
        np.testing.assert_allclose(np.nanmean(out, axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.nanstd(out, axis=0, ddof=1), [1.0, 1.0], atol=1e-12)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="scale argument"):
            scale_mat(np.zeros((2, 2)), "bogus")
