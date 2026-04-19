"""Hierarchical clustering + gap detection.

Ported from ``cluster_mat``, ``find_gaps`` in ``R/pheatmap.r``.

The Python port returns a lightweight ``HClust`` namedtuple whose fields
mirror R's ``hclust`` object components used downstream (``merge``, ``order``,
``height``, ``labels``) plus a ``linkage`` array for use with SciPy's
``fcluster``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, to_tree
from scipy.spatial.distance import pdist, squareform

__all__ = ["HClust", "cluster_mat", "find_gaps", "_hclust_from_linkage"]


_VALID_METHODS = {
    "ward.D", "ward.D2", "ward", "single", "complete", "average",
    "mcquitty", "median", "centroid",
}

_VALID_DISTANCES = {
    "correlation", "euclidean", "maximum", "manhattan", "canberra",
    "binary", "minkowski",
}


# SciPy linkage method name mapping.  ``ward`` in SciPy corresponds to
# R's ``ward.D2``; R's ``ward`` / ``ward.D`` require pre-squaring the input
# dissimilarity matrix, which is not exercised by the tutorial.
_METHOD_TO_SCIPY = {
    "single": "single",
    "complete": "complete",
    "average": "average",
    "mcquitty": "weighted",
    "median": "median",
    "centroid": "centroid",
    "ward.D2": "ward",
    "ward.D": "ward",   # approximation; see note in 05_design.md
    "ward": "ward",     # legacy alias for ward.D
}


# SciPy pdist metric name mapping.
_DISTANCE_TO_SCIPY = {
    "euclidean": "euclidean",
    "maximum": "chebyshev",
    "manhattan": "cityblock",
    "canberra": "canberra",
    "binary": "jaccard",
    "minkowski": "minkowski",
}


@dataclass(frozen=True)
class HClust:
    """Minimal R ``hclust``-like container.

    Attributes
    ----------
    merge
        ``(n-1, 2)`` integer array.  Negative entries reference original
        observations (``-i`` → observation *i*); positive entries reference
        previously formed clusters.  Uses R's 1-based convention so that the
        object round-trips with user-supplied ``hclust`` objects.
    order
        0-based permutation of rows giving the dendrogram leaf order.
    height
        Length ``n - 1`` array of merge heights.
    labels
        Row labels (``None`` if unnamed).
    linkage
        The underlying SciPy linkage matrix (useful for ``fcluster``).
    """

    merge: np.ndarray
    order: np.ndarray
    height: np.ndarray
    labels: Sequence[str] | None
    linkage: np.ndarray

    # Allow downstream code to use hc["order"] style access (R does $order).
    def __getitem__(self, key: str) -> np.ndarray | Sequence[str] | None:
        return getattr(self, key)


def _hclust_from_linkage(
    link: np.ndarray,
    labels: Sequence[str] | None = None,
) -> HClust:
    """Convert a SciPy linkage matrix to an R-style ``HClust``."""
    n = int(link.shape[0] + 1)
    merge = np.zeros((n - 1, 2), dtype=int)
    for i in range(n - 1):
        a, b = int(link[i, 0]), int(link[i, 1])
        merge[i, 0] = -(a + 1) if a < n else (a - n + 1)
        merge[i, 1] = -(b + 1) if b < n else (b - n + 1)
    height = link[:, 2].astype(float).copy()
    order = np.asarray(leaves_list(link), dtype=int)
    return HClust(
        merge=merge,
        order=order,
        height=height,
        labels=list(labels) if labels is not None else None,
        linkage=link,
    )


def cluster_mat(
    mat: np.ndarray,
    distance: Union[str, np.ndarray],
    method: str,
    labels: Sequence[str] | None = None,
) -> HClust:
    """Cluster rows of ``mat`` using a distance/linkage combination.

    Parameters
    ----------
    mat
        Numeric matrix whose rows are the objects to cluster.
    distance
        Either one of ``"correlation"``, ``"euclidean"``, ``"maximum"``,
        ``"manhattan"``, ``"canberra"``, ``"binary"``, ``"minkowski"`` or a
        pre-computed dissimilarity.  A pre-computed distance may be supplied
        as a 1-D condensed vector (as returned by :func:`scipy.spatial.distance.pdist`)
        or as a square 2-D matrix.
    method
        Linkage method (mirrors ``stats::hclust``).
    labels
        Optional row labels.

    Returns
    -------
    HClust
        Lightweight hclust-like container.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            "clustering method has to one form the list: "
            "'ward', 'ward.D', 'ward.D2', 'single', 'complete', 'average', "
            "'mcquitty', 'median' or 'centroid'."
        )

    mat = np.asarray(mat, dtype=float)

    if isinstance(distance, str):
        if distance not in _VALID_DISTANCES:
            raise ValueError(
                "distance has to be a dissimilarity structure as produced by "
                "dist or one measure form the list: 'correlation', "
                "'euclidean', 'maximum', 'manhattan', 'canberra', 'binary', "
                "'minkowski'"
            )
        if distance == "correlation":
            cor = np.corrcoef(mat)
            d_square = 1.0 - cor
            np.fill_diagonal(d_square, 0.0)
            d = squareform(d_square, checks=False)
        else:
            metric = _DISTANCE_TO_SCIPY[distance]
            d = pdist(mat, metric=metric)
    else:
        arr = np.asarray(distance, dtype=float)
        if arr.ndim == 2:
            d = squareform(arr, checks=False)
        else:
            d = arr

    scipy_method = _METHOD_TO_SCIPY[method]
    link = linkage(d, method=scipy_method)
    return _hclust_from_linkage(link, labels=labels)


def find_gaps(tree: HClust, cutree_n: int) -> np.ndarray:
    """Return 0-based positions (in the cluster-ordered vector) where the
    cluster label changes — i.e. where to draw a vertical/horizontal gap.

    Parameters
    ----------
    tree
        An :class:`HClust` object.
    cutree_n
        Target number of clusters.

    Returns
    -------
    np.ndarray
        Array of integer positions.  An empty array means no gaps are drawn
        (e.g. ``cutree_n == 1``).
    """
    clusters = fcluster(tree.linkage, t=cutree_n, criterion="maxclust")
    ordered = clusters[tree.order]
    return np.where(np.diff(ordered) != 0)[0] + 1
