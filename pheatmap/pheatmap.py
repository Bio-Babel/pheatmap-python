"""Top-level :func:`pheatmap` entrypoint.

Ports the 200-line R function at lines 885–1102 of ``R/pheatmap.r``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from grid_py import grid_draw, grid_newpage

from ._breaks import generate_breaks
from ._cluster import HClust, _hclust_from_linkage, cluster_mat, find_gaps
from ._colours import (
    colour_ramp_palette,
    generate_annotation_colours,
    scale_colours,
)
from ._motor import heatmap_motor
from ._pheatmap_object import PHeatmap
from ._scale import scale_mat
from ._utils import identity2, is_na2

__all__ = ["pheatmap"]


# Default colour palette: rev(brewer.pal(7, "RdYlBu")) |> colorRampPalette()(100)
_RDYLBU_7 = [
    "#D73027",
    "#FC8D59",
    "#FEE090",
    "#FFFFBF",
    "#E0F3F8",
    "#91BFDB",
    "#4575B4",
]
_DEFAULT_COLOR = colour_ramp_palette(list(reversed(_RDYLBU_7)))(100)


_ANGLE_LOOKUP = {
    "0": (0.0, 0.5, 1.0),
    "45": (45.0, 1.0, 1.0),
    "90": (90.0, 1.0, 0.5),
    "270": (270.0, 0.0, 0.5),
    "315": (315.0, 0.0, 1.0),
}


def _coerce_dataframe(df: Any) -> pd.DataFrame | None:
    if df is None or is_na2(df):
        return None
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pd.Series):
        return df.to_frame()
    return pd.DataFrame(df)


def _kmeans_rows(mat: np.ndarray, k: int, seed: int = 1245678) -> dict[str, Any]:
    from scipy.cluster.vq import kmeans2

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(mat.shape[0], size=k, replace=False)
    init = mat[init_idx]
    centers, labels = kmeans2(mat, init, iter=100, minit="matrix", seed=seed)
    sizes = np.bincount(labels, minlength=k)
    return {
        "centers": centers,
        "cluster": labels + 1,  # R uses 1-based cluster labels
        "sizes": sizes,
    }


def pheatmap(
    mat: Any,
    color: Sequence[str] | None = None,
    kmeans_k: int | None = None,
    breaks: np.ndarray | None = None,
    border_color: str | None = "grey60",
    cellwidth: float | None = None,
    cellheight: float | None = None,
    scale: str = "none",
    cluster_rows: bool | HClust = True,
    cluster_cols: bool | HClust = True,
    clustering_distance_rows: Any = "euclidean",
    clustering_distance_cols: Any = "euclidean",
    clustering_method: str = "complete",
    clustering_callback: Callable[..., HClust] = identity2,
    cutree_rows: int | None = None,
    cutree_cols: int | None = None,
    treeheight_row: float | None = None,
    treeheight_col: float | None = None,
    legend: bool = True,
    legend_breaks: Sequence[float] | None = None,
    legend_labels: Sequence[str] | None = None,
    annotation_row: pd.DataFrame | None = None,
    annotation_col: pd.DataFrame | None = None,
    annotation: pd.DataFrame | None = None,
    annotation_colors: Mapping[str, Any] | None = None,
    annotation_legend: bool = True,
    annotation_names_row: bool = True,
    annotation_names_col: bool = True,
    drop_levels: bool = True,
    show_rownames: bool = True,
    show_colnames: bool = True,
    main: str | None = None,
    fontsize: float = 10,
    fontsize_row: float | None = None,
    fontsize_col: float | None = None,
    angle_col: str | float = "270",
    display_numbers: Any = False,
    number_format: str = "%.2f",
    number_color: str = "grey30",
    fontsize_number: float | None = None,
    gaps_row: Sequence[int] | None = None,
    gaps_col: Sequence[int] | None = None,
    labels_row: Sequence[str] | None = None,
    labels_col: Sequence[str] | None = None,
    filename: str | None = None,
    width: float | None = None,
    height: float | None = None,
    silent: bool = False,
    na_col: str = "#DDDDDD",
    **kwargs: Any,
) -> PHeatmap:
    """Draw a clustered heatmap.

    Parameters
    ----------
    mat
        Numeric matrix (array-like or :class:`pandas.DataFrame`).
    color
        Vector of colours used in the heatmap; defaults to a 100-step
        reversed ``RdYlBu`` palette.
    kmeans_k
        If set, aggregate rows via k-means into this many clusters before
        drawing.
    breaks
        Colour break points.  If ``None``, computed automatically.
    border_color
        Colour of cell borders; ``None`` disables.
    cellwidth, cellheight
        Cell dimensions in big-points; ``None`` = auto-size to plot window.
    scale
        ``"none"``, ``"row"``, or ``"column"``.
    cluster_rows, cluster_cols
        Either a boolean or a pre-computed :class:`pheatmap._cluster.HClust`.
    clustering_distance_rows, clustering_distance_cols
        Distance measure; passes through :func:`pheatmap._cluster.cluster_mat`.
    clustering_method
        Linkage method (see ``hclust``).
    clustering_callback
        Callable ``(HClust, mat) -> HClust`` for re-ordering the tree.
    cutree_rows, cutree_cols
        If clustering is enabled and this is set, draw gaps between the
        resulting clusters.
    treeheight_row, treeheight_col
        Dendrogram heights (in big-points).
    legend
        Draw colour legend when ``True``.
    legend_breaks, legend_labels
        Explicit tick positions and labels.
    annotation_row, annotation_col, annotation
        Annotation :class:`pandas.DataFrame`.
    annotation_colors
        Mapping ``{column → palette}``.
    annotation_legend, annotation_names_row, annotation_names_col
        Legend / name visibility.
    drop_levels
        Drop unused factor levels from categorical annotations.
    show_rownames, show_colnames
        Label visibility toggles.
    main
        Plot title.
    fontsize, fontsize_row, fontsize_col, fontsize_number
        Font sizes (points).
    angle_col
        One of ``"0", "45", "90", "270", "315"``.
    display_numbers
        ``True``/``False``/matrix-of-strings.
    number_format, number_color
        Formatting for on-cell numbers.
    gaps_row, gaps_col
        Manual gap positions (1-based, in the *final* row/column order).
    labels_row, labels_col
        Override row/column labels.
    filename, width, height
        File output (disables on-screen draw).
    silent
        If ``True``, do not draw to the current device.
    na_col
        Colour for NaN cells.

    Returns
    -------
    PHeatmap
        Container with ``tree_row``, ``tree_col``, ``kmeans``, ``gtable``.
    """
    # Accept legacy partial-match `cluster_row` as alias for `cluster_rows`.
    if "cluster_row" in kwargs:
        cluster_rows = kwargs.pop("cluster_row")
    if "cluster_col" in kwargs:
        cluster_cols = kwargs.pop("cluster_col")
    if "cutree_col" in kwargs:
        cutree_cols = kwargs.pop("cutree_col")
    if "cutree_row" in kwargs:
        cutree_rows = kwargs.pop("cutree_row")

    if color is None:
        color = list(_DEFAULT_COLOR)
    color = list(color)

    if fontsize_row is None:
        fontsize_row = fontsize
    if fontsize_col is None:
        fontsize_col = fontsize
    if fontsize_number is None:
        fontsize_number = 0.8 * fontsize
    if treeheight_row is None:
        treeheight_row = 50.0 if (isinstance(cluster_rows, HClust) or bool(cluster_rows)) else 0.0
    if treeheight_col is None:
        treeheight_col = 50.0 if (isinstance(cluster_cols, HClust) or bool(cluster_cols)) else 0.0

    # Set default labels from matrix names.
    if isinstance(mat, pd.DataFrame):
        if labels_row is None:
            labels_row = list(mat.index.astype(str))
        if labels_col is None:
            labels_col = list(mat.columns.astype(str))
        mat_arr = mat.values.astype(float)
    else:
        mat_arr = np.asarray(mat, dtype=float)
        if labels_row is None:
            labels_row = None
        if labels_col is None:
            labels_col = None
    labels_row = list(labels_row) if labels_row is not None else None
    labels_col = list(labels_col) if labels_col is not None else None

    # Preprocess matrix: scale.
    mat_arr = np.asarray(mat_arr, dtype=float)
    if scale != "none":
        mat_arr = scale_mat(mat_arr, scale)
        if breaks is None or is_na2(breaks):
            breaks = generate_breaks(mat_arr, len(color), center=True)

    # K-means row aggregation.
    km_result = None
    if kmeans_k is not None and not is_na2(kmeans_k):
        km_result = _kmeans_rows(mat_arr, int(kmeans_k))
        mat_arr = np.asarray(km_result["centers"], dtype=float)
        sizes = km_result["sizes"]
        labels_row = [f"Cluster: {i+1} Size: {sizes[i]}" for i in range(len(sizes))]

    # Format display numbers.
    draw_numbers = False
    if isinstance(display_numbers, (np.ndarray, pd.DataFrame)):
        disp = (
            display_numbers.values if isinstance(display_numbers, pd.DataFrame)
            else np.asarray(display_numbers)
        )
        if disp.shape != mat_arr.shape:
            raise ValueError(
                "If display_numbers provided as matrix, its dimensions have to "
                "match with mat"
            )
        fmat = np.asarray(disp, dtype=object).astype(str)
        draw_numbers = True
    else:
        if bool(display_numbers):
            fmat = np.empty(mat_arr.shape, dtype=object)
            it = np.nditer(mat_arr, flags=["multi_index"])
            while not it.finished:
                fmat[it.multi_index] = number_format % float(it[0])
                it.iternext()
            draw_numbers = True
        else:
            fmat = np.full(mat_arr.shape, "", dtype=object)
            draw_numbers = False

    # Row clustering.
    if isinstance(cluster_rows, HClust) or bool(cluster_rows):
        if isinstance(cluster_rows, HClust):
            tree_row = cluster_rows
        else:
            tree_row = cluster_mat(
                mat_arr,
                distance=clustering_distance_rows,
                method=clustering_method,
                labels=labels_row,
            )
            tree_row = clustering_callback(tree_row, mat_arr)
        order_r = np.asarray(tree_row.order, dtype=int)
        mat_arr = mat_arr[order_r, :]
        fmat = fmat[order_r, :]
        if labels_row is not None:
            labels_row = [labels_row[i] for i in order_r]
        if cutree_rows is not None and not is_na2(cutree_rows):
            gaps_row = list(find_gaps(tree_row, int(cutree_rows)))
        else:
            gaps_row = None
    else:
        tree_row = None
        treeheight_row = 0.0

    # Col clustering.
    if isinstance(cluster_cols, HClust) or bool(cluster_cols):
        if isinstance(cluster_cols, HClust):
            tree_col = cluster_cols
        else:
            tree_col = cluster_mat(
                mat_arr.T,
                distance=clustering_distance_cols,
                method=clustering_method,
                labels=labels_col,
            )
            tree_col = clustering_callback(tree_col, mat_arr.T)
        order_c = np.asarray(tree_col.order, dtype=int)
        mat_arr = mat_arr[:, order_c]
        fmat = fmat[:, order_c]
        if labels_col is not None:
            labels_col = [labels_col[i] for i in order_c]
        if cutree_cols is not None and not is_na2(cutree_cols):
            gaps_col = list(find_gaps(tree_col, int(cutree_cols)))
        else:
            gaps_col = None
    else:
        tree_col = None
        treeheight_col = 0.0

    # Colours and scales.
    if (
        legend_breaks is not None
        and legend_labels is not None
        and not is_na2(legend_breaks)
        and not is_na2(legend_labels)
    ):
        if len(legend_breaks) != len(legend_labels):
            raise ValueError(
                "Lengths of legend_breaks and legend_labels must be the same"
            )

    if breaks is None or is_na2(breaks):
        breaks = generate_breaks(mat_arr, len(color))
    breaks = np.asarray(breaks, dtype=float)

    # R: independently augment with -Inf at the front and +Inf at the back.
    # The previous outer guard short-circuited incorrectly, and `np.isinf`
    # also matched -Inf at the tail.
    if not np.isneginf(np.min(breaks)):
        breaks = np.concatenate([[-np.inf], breaks])
        color = [color[0]] + color
    if not np.isposinf(np.max(breaks)):
        breaks = np.concatenate([breaks, [np.inf]])
        color = color + [color[-1]]
    non_inf_breaks = breaks[np.isfinite(breaks)]

    if legend and (legend_breaks is None or is_na2(legend_breaks)):
        lo_hi = (float(non_inf_breaks.min()), float(non_inf_breaks.max()))
        legend_vals = list(np.asarray(__import__("grid_py").grid_pretty(lo_hi), dtype=float))
        legend_dict = {str(v): v for v in legend_vals}
    elif legend and legend_breaks is not None and not is_na2(legend_breaks):
        lb = np.asarray(list(legend_breaks), dtype=float)
        mask = (lb >= non_inf_breaks.min()) & (lb <= non_inf_breaks.max())
        lb_filt = lb[mask]
        if legend_labels is not None and not is_na2(legend_labels):
            ll = np.asarray(list(legend_labels), dtype=object)[mask]
            legend_dict = {str(lab): float(val) for lab, val in zip(ll, lb_filt)}
        else:
            legend_dict = {str(v): float(v) for v in lb_filt}
    else:
        legend_dict = None

    # Re-colour matrix via breaks.
    colour_matrix = scale_colours(mat_arr, col=color, breaks=breaks, na_col=na_col)

    # Annotations.
    annotation_col = _coerce_dataframe(annotation_col)
    annotation_row = _coerce_dataframe(annotation_row)
    annotation = _coerce_dataframe(annotation)
    if annotation_col is None and annotation is not None:
        import warnings

        warnings.warn(
            "'annotation' is deprecated; use 'annotation_col' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        annotation_col = annotation

    # R: annotation_col[colnames(mat), , drop=F] — preserves matrix column
    # order and yields NA-filled rows for any name missing in the annotation.
    if annotation_col is not None and labels_col is not None:
        annotation_col = annotation_col.reindex(list(labels_col))
    if annotation_row is not None and labels_row is not None:
        annotation_row = annotation_row.reindex(list(labels_row))

    ann_map: dict[str, pd.Series] = {}
    if annotation_row is not None:
        ann_map.update({c: annotation_row[c] for c in annotation_row.columns})
    if annotation_col is not None:
        ann_map.update({c: annotation_col[c] for c in annotation_col.columns})

    if len(ann_map) > 0:
        annotation_colors = generate_annotation_colours(
            ann_map, annotation_colors, drop=drop_levels
        )
    else:
        annotation_colors = None

    if not show_rownames:
        labels_row = None
    if not show_colnames:
        labels_col = None

    ac = str(angle_col)
    if ac not in _ANGLE_LOOKUP:
        raise ValueError(
            "angle_col must be one of '0', '45', '90', '270', '315'"
        )
    rot, hjust_col, vjust_col = _ANGLE_LOOKUP[ac]

    gt = heatmap_motor(
        colour_matrix,
        border_color=border_color,
        cellwidth=cellwidth,
        cellheight=cellheight,
        tree_col=tree_col,
        tree_row=tree_row,
        treeheight_col=treeheight_col,
        treeheight_row=treeheight_row,
        filename=filename,
        width=width,
        height=height,
        breaks=breaks,
        color=color,
        legend=legend_dict,
        annotation_row=annotation_row,
        annotation_col=annotation_col,
        annotation_colors=annotation_colors,
        annotation_legend=annotation_legend,
        annotation_names_row=annotation_names_row,
        annotation_names_col=annotation_names_col,
        main=main,
        fontsize=fontsize,
        fontsize_row=fontsize_row,
        fontsize_col=fontsize_col,
        hjust_col=hjust_col,
        vjust_col=vjust_col,
        angle_col=rot,
        fmat=fmat,
        fmat_draw=draw_numbers,
        fontsize_number=fontsize_number,
        number_color=number_color,
        gaps_col=gaps_col,
        gaps_row=gaps_row,
        labels_row=labels_row,
        labels_col=labels_col,
        **kwargs,
    )

    result = PHeatmap(
        tree_row=tree_row, tree_col=tree_col, kmeans=km_result, gtable=gt
    )

    if (filename is None or is_na2(filename)) and not silent:
        try:
            grid_newpage()
            grid_draw(gt)
        except Exception:
            pass

    return result
