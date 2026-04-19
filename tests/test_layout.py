"""Tests for :mod:`pheatmap._layout`."""

from __future__ import annotations

import numpy as np
import pytest
from grid_py import convert_width, grid_newpage

from pheatmap._layout import find_coordinates


@pytest.fixture(autouse=True)
def _open_grid_device() -> None:
    """Ensure a grid device exists so ``1npc`` resolves to a concrete size."""
    grid_newpage()


def _to_bigpts(unit) -> np.ndarray:
    """Resolve an arbitrary :class:`Unit` (potentially a sum unit) to bigpts."""
    v = convert_width(unit, "bigpts", valueOnly=True)
    return np.atleast_1d(np.asarray(v, dtype=float))


class TestFindCoordinates:
    def test_no_gaps_uses_npc(self) -> None:
        res = find_coordinates(5, gaps=None)
        assert res["size"] is not None
        assert res["coord"] is not None

    def test_rejects_gaps_past_end(self) -> None:
        with pytest.raises(ValueError, match="Gaps do not match"):
            find_coordinates(5, gaps=[10])

    def test_coord_length_matches_n(self) -> None:
        res = find_coordinates(7, gaps=None)
        vals = np.atleast_1d(np.asarray(res["coord"].values, dtype=float))
        assert len(vals) == 7

    def test_coord_with_gaps_increases_monotonically(self) -> None:
        res = find_coordinates(8, gaps=[3, 6])
        vals = _to_bigpts(res["coord"])
        assert np.all(np.diff(vals) > 0)
        assert len(vals) == 8
