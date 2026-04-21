"""
pheatmap_py — Python port of the R ``pheatmap`` package (v1.0.13).

The public surface mirrors R's single exported function :func:`pheatmap` plus
its two S3 methods (:func:`grid_draw` / :meth:`PHeatmap.draw`,
:meth:`PHeatmap._ipython_display_`).

Internal helpers (``scale_mat``, ``cluster_mat``, ``generate_breaks``, …) are
also re-exported so users can build heatmaps from intermediate objects.
"""

from __future__ import annotations

from ._breaks import generate_breaks
from ._cluster import HClust, cluster_mat, find_gaps
from ._colours import (
    colorRampPalette,
    colour_ramp_palette,
    convert_annotations,
    generate_annotation_colours,
    scale_colours,
    scale_vec_colours,
)
from ._grobs import (
    draw_annotation_legend,
    draw_annotation_names,
    draw_annotations,
    draw_colnames,
    draw_dendrogram,
    draw_legend,
    draw_main,
    draw_matrix,
    draw_rownames,
)
from ._layout import find_coordinates, lo
from ._motor import heatmap_motor
from ._pheatmap_object import PHeatmap, grid_draw
from ._scale import scale_mat, scale_rows
from ._utils import identity2, is_na2
from .pheatmap import pheatmap

__version__ = "1.0.13"
__r_commit__ = "ffd0f8c"

__all__ = [
    "pheatmap",
    "PHeatmap",
    "grid_draw",
    # internal helpers surfaced for advanced users
    "HClust",
    "cluster_mat",
    "find_gaps",
    "generate_breaks",
    "scale_mat",
    "scale_rows",
    "scale_colours",
    "scale_vec_colours",
    "convert_annotations",
    "generate_annotation_colours",
    "colour_ramp_palette",
    "colorRampPalette",
    "is_na2",
    "identity2",
    "lo",
    "find_coordinates",
    "heatmap_motor",
    "draw_matrix",
    "draw_colnames",
    "draw_rownames",
    "draw_dendrogram",
    "draw_legend",
    "draw_annotations",
    "draw_annotation_names",
    "draw_annotation_legend",
    "draw_main",
]
