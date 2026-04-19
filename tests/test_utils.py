"""Tests for :mod:`pheatmap._utils`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pheatmap._utils import identity2, is_na2


class TestIsNa2:
    def test_none_is_true(self) -> None:
        assert is_na2(None) is True

    def test_na_scalar(self) -> None:
        assert is_na2(np.nan) is True
        assert is_na2(pd.NA) is True

    def test_finite_scalar(self) -> None:
        assert is_na2(0.0) is False
        assert is_na2(1) is False
        assert is_na2("hello") is False

    def test_array_not_scalar(self) -> None:
        # R's is.na2 returns FALSE for non-empty arrays even if they contain NaN.
        assert is_na2(np.array([np.nan, 1.0])) is False

    def test_empty_array_is_na(self) -> None:
        # Matches the length-0 branch of is_na2.
        assert is_na2(np.array([])) is True

    def test_dataframe_is_not_na(self) -> None:
        # DataFrames carry their own missingness info so they always count
        # as "something" to is_na2.
        assert is_na2(pd.DataFrame({"a": [1]})) is False
        assert is_na2(pd.DataFrame()) is False


class TestIdentity2:
    def test_passthrough_with_extra_args(self) -> None:
        sentinel = object()
        assert identity2(sentinel, "ignored", kw="ignored") is sentinel
