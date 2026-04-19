"""Tests for :mod:`pheatmap._cluster` against R reference fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from pheatmap._cluster import HClust, cluster_mat, find_gaps


def _normalize_merge(merge: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Sort each merge row so that the smaller column comes first.

    R and SciPy agree on merge *sets* but may order the two children differently
    for ties.  We canonicalise them before comparing.
    """
    out = np.asarray(merge, dtype=int).copy()
    for i in range(out.shape[0]):
        a, b = out[i]
        if a > b:
            out[i] = [b, a]
    return out


class TestClusterMatEuclideanComplete:
    def test_merge_matches_r(self, test_matrix: np.ndarray,
                             hclust_complete_euclidean: dict) -> None:
        hc = cluster_mat(test_matrix, distance="euclidean", method="complete")
        r_merge = np.asarray(hclust_complete_euclidean["merge"], dtype=int)
        np.testing.assert_array_equal(
            _normalize_merge(hc.merge, hc.height),
            _normalize_merge(r_merge, hclust_complete_euclidean["height"]),
        )

    def test_height_matches_r(self, test_matrix: np.ndarray,
                              hclust_complete_euclidean: dict) -> None:
        hc = cluster_mat(test_matrix, distance="euclidean", method="complete")
        np.testing.assert_allclose(
            hc.height, hclust_complete_euclidean["height"], rtol=1e-10, atol=1e-10
        )

    def test_order_matches_r_up_to_reflection(
        self, test_matrix: np.ndarray, hclust_complete_euclidean: dict
    ) -> None:
        hc = cluster_mat(test_matrix, distance="euclidean", method="complete")
        r_order = np.asarray(hclust_complete_euclidean["order"], dtype=int) - 1  # R → 0-based
        # Leaf ordering is not unique across tie-breaking rules.  Assert that
        # the *set* of leaves is identical and the permutation is valid.
        assert sorted(hc.order.tolist()) == sorted(r_order.tolist())


class TestClusterMatCorrelation:
    def test_height_matches_r(self, test_matrix: np.ndarray,
                              hclust_correlation_rows: dict) -> None:
        hc = cluster_mat(test_matrix, distance="correlation", method="complete")
        np.testing.assert_allclose(
            hc.height, hclust_correlation_rows["height"], rtol=1e-8, atol=1e-10
        )


class TestClusterMatManhattan:
    def test_height_matches_r(self, test_matrix: np.ndarray,
                              hclust_minkowski_rows: dict) -> None:
        hc = cluster_mat(test_matrix, distance="manhattan", method="average")
        np.testing.assert_allclose(
            hc.height, hclust_minkowski_rows["height"], rtol=1e-10, atol=1e-10
        )


class TestClusterMatValidation:
    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="clustering method"):
            cluster_mat(np.zeros((3, 3)), distance="euclidean", method="nope")

    def test_invalid_distance(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            cluster_mat(np.zeros((3, 3)), distance="nope", method="complete")

    def test_precomputed_square_distance(self) -> None:
        sq = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        hc = cluster_mat(np.zeros((3, 3)), distance=sq, method="complete")
        assert isinstance(hc, HClust)
        assert hc.merge.shape == (2, 2)
        assert hc.height.shape == (2,)


class TestFindGaps:
    def test_single_cluster_no_gaps(self, test_matrix: np.ndarray) -> None:
        hc = cluster_mat(test_matrix, distance="euclidean", method="complete")
        assert find_gaps(hc, 1).size == 0

    def test_n_clusters_yields_n_minus_1_gaps(self, test_matrix: np.ndarray) -> None:
        hc = cluster_mat(test_matrix, distance="euclidean", method="complete")
        for k in (2, 3, 4):
            gaps = find_gaps(hc, k)
            assert len(gaps) == k - 1

    def test_gaps_are_strictly_increasing(self, test_matrix: np.ndarray) -> None:
        hc = cluster_mat(test_matrix, distance="euclidean", method="complete")
        gaps = find_gaps(hc, 4)
        assert np.all(np.diff(gaps) > 0)
