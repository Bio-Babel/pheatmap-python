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


class TestUseRaster:
    """``use_raster=True`` embeds the heatmap body as a single raster image,
    matching ``ComplexHeatmap::Heatmap(use_raster = TRUE)``.  The motivation is
    PDF file size on large matrices.
    """

    def test_returns_valid_pheatmap_with_raster(
        self, test_matrix: np.ndarray
    ) -> None:
        res = pheatmap(test_matrix, use_raster=True, silent=True)
        assert isinstance(res, PHeatmap)
        assert res.gtable is not None

    def test_raster_pdf_far_smaller_than_vector(self, tmp_path) -> None:
        rng = np.random.RandomState(0)
        big = rng.randn(50, 1500)  # 75 000 cells: enough to show the win
        vec = tmp_path / "vec.pdf"
        ras = tmp_path / "ras.pdf"
        pheatmap(big, filename=str(vec), width=6, height=4,
                 silent=True, use_raster=False)
        pheatmap(big, filename=str(ras), width=6, height=4,
                 silent=True, use_raster=True)
        # Raster path drops one rect per cell. At 75 000 cells the fixed
        # overhead (dendrograms, legend, frame) still dominates, so we only
        # assert a 3x win here; at copykat scale (~3M cells) the win is ~50x.
        assert ras.stat().st_size * 3 < vec.stat().st_size, (
            f"raster {ras.stat().st_size} B vs vector {vec.stat().st_size} B"
        )

    def test_raster_chunked_with_gaps(self, test_matrix: np.ndarray) -> None:
        """``gaps_col`` / ``gaps_row`` split the matrix body into multiple
        raster tiles, one per chunk (mirrors ``ComplexHeatmap`` behaviour for
        split heatmaps). The call succeeds and the gtable is populated."""
        rows, cols = test_matrix.shape
        gr, gc = [rows // 2], [cols // 3, 2 * cols // 3]
        res = pheatmap(
            test_matrix,
            cluster_rows=False,
            cluster_cols=False,
            gaps_col=gc,
            gaps_row=gr,
            use_raster=True,
            silent=True,
        )
        assert res.gtable is not None

    def test_raster_gaps_emit_one_grob_per_chunk(self, test_matrix: np.ndarray) -> None:
        """Each (row_chunk x col_chunk) cross product produces one raster grob."""
        from pheatmap._grobs import _chunk_ranges, _draw_matrix_raster

        rows, cols = test_matrix.shape
        # Bypass the full pheatmap pipeline by passing a literal colour matrix.
        colour_mat = np.full(test_matrix.shape, "#67a9cf", dtype=object)
        gaps_rows, gaps_cols = [rows // 2], [cols // 3, 2 * cols // 3]
        tree = _draw_matrix_raster(
            colour_mat, gaps_rows, gaps_cols,
            fmat=np.zeros_like(colour_mat),
            fontsize_number=10, number_color="grey30",
            draw_numbers=False, interpolate=False,
        )
        expected = len(_chunk_ranges(rows, gaps_rows)) * len(_chunk_ranges(cols, gaps_cols))
        children = getattr(tree, "children", None) or getattr(tree, "_children", None)
        assert children is not None and len(children) == expected, (
            f"expected {expected} raster grobs, got {len(children) if children else 'None'}"
        )

    def test_chunk_ranges_skips_empty(self) -> None:
        """A trailing gap at position ``n`` produces no empty chunk."""
        from pheatmap._grobs import _chunk_ranges
        assert _chunk_ranges(10, None) == [(0, 10)]
        assert _chunk_ranges(10, []) == [(0, 10)]
        assert _chunk_ranges(10, [3, 7]) == [(0, 3), (3, 7), (7, 10)]
        assert _chunk_ranges(10, [5, 10]) == [(0, 5), (5, 10)]   # trailing skipped
        assert _chunk_ranges(10, [0, 5]) == [(0, 5), (5, 10)]    # leading skipped
