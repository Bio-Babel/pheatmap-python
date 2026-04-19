"""Tests for :mod:`pheatmap._colours`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pheatmap._colours import (
    colour_ramp_palette,
    convert_annotations,
    generate_annotation_colours,
    r_style_cut,
    scale_colours,
    scale_vec_colours,
)


class TestColourRampPalette:
    def test_endpoints_preserved(self) -> None:
        ramp = colour_ramp_palette(["#000000", "#FFFFFF"])
        out = ramp(2)
        assert out[0].upper().startswith("#0")
        assert out[-1].upper().startswith("#F")

    def test_length(self) -> None:
        ramp = colour_ramp_palette(["red", "blue"])
        assert len(ramp(7)) == 7

    def test_zero_returns_empty(self) -> None:
        assert colour_ramp_palette(["red", "blue"])(0) == []


class TestRStyleCut:
    def test_include_lowest(self) -> None:
        breaks = np.linspace(0.0, 1.0, 5)  # 0, .25, .5, .75, 1
        # The value equal to the lowest break maps to bin 0, matching
        # include.lowest=TRUE in R.
        idx = r_style_cut(np.array([0.0]), breaks)
        assert idx[0] == 0

    def test_upper_bound_included(self) -> None:
        breaks = np.linspace(0.0, 1.0, 5)
        idx = r_style_cut(np.array([1.0]), breaks)
        assert idx[0] == 3  # last bin index = len(breaks) - 2

    def test_nan_maps_to_minus_one(self) -> None:
        idx = r_style_cut(np.array([np.nan]), np.array([0.0, 1.0]))
        assert idx[0] == -1


class TestScaleVecColours:
    def test_vec_matches_expected(self) -> None:
        col = ["#000000", "#808080", "#FFFFFF"]
        breaks = np.linspace(0.0, 1.0, 4)  # 3 bins
        out = scale_vec_colours(np.array([0.0, 0.5, 1.0]), col, breaks, na_col="#FF0000")
        assert list(out) == ["#000000", "#808080", "#FFFFFF"]

    def test_na_falls_back_to_na_col(self) -> None:
        out = scale_vec_colours(
            np.array([np.nan]),
            ["#000000", "#FFFFFF"],
            breaks=np.array([0.0, 1.0]),
            na_col="#00FF00",
        )
        assert out[0] == "#00FF00"


class TestScaleColoursMatrix:
    def test_preserves_shape(self) -> None:
        m = np.array([[0.0, 0.5], [0.5, 1.0]])
        out = scale_colours(
            m,
            col=["#000000", "#FFFFFF"],
            breaks=np.linspace(0.0, 1.0, 3),
            na_col="#FF0000",
        )
        assert out.shape == m.shape


class TestConvertAnnotations:
    def test_categorical_mapping(self) -> None:
        df = pd.DataFrame({"grp": ["A", "B", "A"]})
        colors = {"grp": {"A": "#111111", "B": "#222222"}}
        out = convert_annotations(df, colors)
        assert list(out[:, 0]) == ["#111111", "#222222", "#111111"]

    def test_unknown_level_raises(self) -> None:
        df = pd.DataFrame({"grp": ["A", "C"]})
        colors = {"grp": {"A": "#111111", "B": "#222222"}}
        with pytest.raises(ValueError, match="Factor levels"):
            convert_annotations(df, colors)

    def test_continuous_produces_colours(self) -> None:
        df = pd.DataFrame({"score": [0.0, 1.0, 2.0]})
        colors = {"score": ["#FFFFFF", "#000000"]}
        out = convert_annotations(df, colors)
        assert all(isinstance(v, str) and v.startswith("#") for v in out[:, 0])
        # Low value should be closer to first colour than high value.
        assert out[0, 0] != out[-1, 0]


class TestGenerateAnnotationColours:
    def test_assigns_palette_for_missing_categoricals(self) -> None:
        df = pd.DataFrame({"grp": ["A", "B", "C"]})
        colors = generate_annotation_colours(df, annotation_colors=None)
        assert "grp" in colors
        assert set(colors["grp"].keys()) == {"A", "B", "C"}
        assert all(isinstance(v, str) and v.startswith("#") for v in colors["grp"].values())

    def test_preserves_user_supplied_colours(self) -> None:
        df = pd.DataFrame({"grp": ["A", "B"]})
        user = {"grp": {"A": "#111111", "B": "#222222"}}
        out = generate_annotation_colours(df, annotation_colors=user)
        assert out["grp"] == user["grp"]

    def test_continuous_gets_list_palette(self) -> None:
        df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 5)})
        out = generate_annotation_colours(df, annotation_colors=None)
        assert isinstance(out["x"], list)
        assert len(out["x"]) == 4
