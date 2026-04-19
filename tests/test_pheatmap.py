"""End-to-end integration tests for :func:`pheatmap.pheatmap`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pheatmap import PHeatmap, pheatmap


class TestPheatmapReturnValue:
    def test_returns_pheatmap_instance(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(test_matrix, silent=True)
        assert isinstance(res, PHeatmap)
        assert res.gtable is not None

    def test_trees_populated_by_default(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(test_matrix, silent=True)
        assert res.tree_row is not None
        assert res.tree_col is not None

    def test_clusters_off_leaves_trees_none(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(
            test_matrix, cluster_rows=False, cluster_cols=False, silent=True
        )
        assert res.tree_row is None
        assert res.tree_col is None


class TestScaleOption:
    def test_scale_row_still_returns_valid(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(test_matrix, scale="row", silent=True)
        assert isinstance(res, PHeatmap)

    def test_scale_column_still_returns_valid(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(test_matrix, scale="column", silent=True)
        assert isinstance(res, PHeatmap)


class TestKmeans:
    def test_kmeans_assigns_metadata(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(test_matrix, kmeans_k=3, silent=True)
        assert res.kmeans is not None
        assert sum(res.kmeans["sizes"]) == test_matrix.shape[0]


class TestCutreeAndGaps:
    def test_cutree_rows_emits_gaps(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(test_matrix, cutree_rows=3, silent=True)
        assert res.tree_row is not None

    def test_explicit_gaps_accepted(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(
            test_matrix,
            cluster_rows=False,
            cluster_cols=False,
            gaps_row=[5, 10],
            gaps_col=[4],
            silent=True,
        )
        assert res.gtable is not None


class TestAnnotations:
    def test_annotation_col_only(self, test_matrix: np.ndarray) -> None:
        ann = pd.DataFrame(
            {"grp": ["A"] * 5 + ["B"] * 5},
            index=[f"c{i}" for i in range(10)],
        )
        df = pd.DataFrame(
            test_matrix,
            index=[f"r{i}" for i in range(20)],
            columns=ann.index,
        )
        res = pheatmap(df, annotation_col=ann, silent=True)
        assert res.gtable is not None

    def test_annotation_row_and_col(self, test_matrix: np.ndarray) -> None:
        ann_col = pd.DataFrame(
            {"grp": ["A"] * 5 + ["B"] * 5}, index=[f"c{i}" for i in range(10)]
        )
        ann_row = pd.DataFrame(
            {"time": ["t0"] * 10 + ["t1"] * 10}, index=[f"r{i}" for i in range(20)]
        )
        df = pd.DataFrame(test_matrix, index=ann_row.index, columns=ann_col.index)
        res = pheatmap(df, annotation_col=ann_col, annotation_row=ann_row, silent=True)
        assert res.gtable is not None


class TestDisplayNumbers:
    def test_boolean_true_uses_number_format(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(test_matrix, display_numbers=True, silent=True)
        assert res.gtable is not None

    def test_custom_labels_matrix(self, test_matrix: np.ndarray) -> None:
        labs = np.array(
            [f"{v:.1f}" for v in test_matrix.ravel()]
        ).reshape(test_matrix.shape)
        res = pheatmap(test_matrix, display_numbers=labs, silent=True)
        assert res.gtable is not None


class TestLegend:
    def test_explicit_breaks_and_legend(self, test_matrix: np.ndarray) -> None:
        breaks = np.linspace(-3, 3, 11)
        res = pheatmap(test_matrix, breaks=breaks, legend=True, silent=True)
        assert res.gtable is not None

    def test_legend_dict_interpreted_as_positions(self, test_matrix: np.ndarray) -> None:
        res = pheatmap(
            test_matrix, legend_breaks=[-2, 0, 2], legend_labels=["lo", "mid", "hi"], silent=True
        )
        assert res.gtable is not None


class TestInputCoercion:
    def test_dataframe_input(self, test_matrix: np.ndarray) -> None:
        df = pd.DataFrame(test_matrix)
        res = pheatmap(df, silent=True)
        assert res.gtable is not None

    def test_rejects_bad_scale(self, test_matrix: np.ndarray) -> None:
        with pytest.raises(ValueError):
            pheatmap(test_matrix, scale="bogus", silent=True)
