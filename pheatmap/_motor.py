"""Assemble a populated :class:`gtable_py.Gtable` from the heatmap
inputs.  Ported from ``heatmap_motor`` in ``R/pheatmap.r``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from grid_py import Gpar, grid_draw, grid_newpage
from gtable_py import gtable_add_grob, gtable_height, gtable_width

from ._colours import convert_annotations
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
from ._layout import lo
from ._utils import is_na2

__all__ = ["heatmap_motor"]


def _annotation_series_map(df: pd.DataFrame | None) -> dict[str, pd.Series]:
    if df is None or is_na2(df):
        return {}
    return {col: df[col] for col in df.columns}


def heatmap_motor(
    matrix: np.ndarray,
    border_color: str | None,
    cellwidth: float | None,
    cellheight: float | None,
    tree_col: Any,
    tree_row: Any,
    treeheight_col: float,
    treeheight_row: float,
    filename: str | None,
    width: float | None,
    height: float | None,
    breaks: np.ndarray,
    color: Sequence[str],
    legend: Any,
    annotation_row: pd.DataFrame | None,
    annotation_col: pd.DataFrame | None,
    annotation_colors: Mapping[str, Any] | None,
    annotation_legend: bool,
    annotation_names_row: bool,
    annotation_names_col: bool,
    main: str | None,
    fontsize: float,
    fontsize_row: float,
    fontsize_col: float,
    hjust_col: float,
    vjust_col: float,
    angle_col: float,
    fmat: np.ndarray,
    fmat_draw: bool,
    fontsize_number: float,
    number_color: str,
    gaps_col: Sequence[int] | None,
    gaps_row: Sequence[int] | None,
    labels_row: Sequence[str] | None,
    labels_col: Sequence[str] | None,
    **kwargs: Any,
):
    layout = lo(
        rown=labels_row,
        coln=labels_col,
        nrow=matrix.shape[0],
        ncol=matrix.shape[1],
        cellwidth=cellwidth,
        cellheight=cellheight,
        treeheight_col=treeheight_col,
        treeheight_row=treeheight_row,
        legend=legend,
        annotation_col=annotation_col,
        annotation_row=annotation_row,
        annotation_colors=annotation_colors,
        annotation_legend=annotation_legend,
        annotation_names_row=annotation_names_row,
        annotation_names_col=annotation_names_col,
        main=main,
        fontsize=fontsize,
        fontsize_row=fontsize_row,
        fontsize_col=fontsize_col,
        angle_col=angle_col,
        gaps_row=gaps_row,
        gaps_col=gaps_col,
        **kwargs,
    )
    res = layout["gt"]
    mindim = layout["mindim"]

    # Omit border when the cells are very small.
    if mindim < 3:
        border_color = None

    # Main title (row 1, col 3).
    if main is not None and not is_na2(main):
        elem = draw_main(main, fontsize=1.3 * fontsize, **kwargs)
        res = gtable_add_grob(res, elem, t=1, l=3, name="main", clip="off")

    # Col dendrogram (row 2, col 3).
    if not is_na2(tree_col) and treeheight_col != 0:
        elem = draw_dendrogram(tree_col, gaps_col, horizontal=True)
        res = gtable_add_grob(res, elem, t=2, l=3, name="col_tree")

    # Row dendrogram (row 4, col 1).
    if not is_na2(tree_row) and treeheight_row != 0:
        elem = draw_dendrogram(tree_row, gaps_row, horizontal=False)
        res = gtable_add_grob(res, elem, t=4, l=1, name="row_tree")

    # Matrix (row 4, col 3).
    elem = draw_matrix(
        matrix,
        border_color=border_color,
        gaps_rows=gaps_row,
        gaps_cols=gaps_col,
        fmat=fmat,
        fontsize_number=fontsize_number,
        number_color=number_color,
        draw_numbers=fmat_draw,
    )
    res = gtable_add_grob(res, elem, t=4, l=3, clip="off", name="matrix")

    if labels_col is not None and len(labels_col) != 0:
        elem = draw_colnames(
            labels_col,
            gaps=gaps_col,
            fontsize=fontsize_col,
            hjust_col=hjust_col,
            vjust_col=vjust_col,
            angle_col=angle_col,
            **kwargs,
        )
        res = gtable_add_grob(res, elem, t=5, l=3, clip="off", name="col_names")

    if labels_row is not None and len(labels_row) != 0:
        elem = draw_rownames(labels_row, gaps=gaps_row, fontsize=fontsize_row, **kwargs)
        res = gtable_add_grob(res, elem, t=4, l=4, clip="off", name="row_names")

    # Column annotations (row 3, col 3).
    if not is_na2(annotation_col):
        conv = convert_annotations(annotation_col, annotation_colors or {})
        elem = draw_annotations(conv, border_color, gaps_col, fontsize, horizontal=True)
        res = gtable_add_grob(res, elem, t=3, l=3, clip="off", name="col_annotation")
        if annotation_names_col:
            elem = draw_annotation_names(annotation_col, fontsize, horizontal=True)
            res = gtable_add_grob(res, elem, t=3, l=4, clip="off", name="col_annotation_names")

    # Row annotations (row 4, col 2).
    if not is_na2(annotation_row):
        conv = convert_annotations(annotation_row, annotation_colors or {})
        elem = draw_annotations(conv, border_color, gaps_row, fontsize, horizontal=False)
        res = gtable_add_grob(res, elem, t=4, l=2, clip="off", name="row_annotation")
        if annotation_names_row:
            elem = draw_annotation_names(
                annotation_row,
                fontsize,
                horizontal=False,
                hjust_col=hjust_col,
                vjust_col=vjust_col,
                angle_col=angle_col,
            )
            res = gtable_add_grob(
                res, elem, t=5, l=2, clip="off", name="row_annotation_names"
            )

    # Annotation legend (row 3 or 4, col 6).
    ann_col_map = _annotation_series_map(annotation_col)
    ann_row_map = _annotation_series_map(annotation_row)
    # R concatenates reverse(annotation_col) then reverse(annotation_row).
    annotation_all = dict(reversed(list(ann_col_map.items())))
    annotation_all.update(dict(reversed(list(ann_row_map.items()))))

    if len(annotation_all) > 0 and annotation_legend:
        elem = draw_annotation_legend(
            annotation_all, annotation_colors or {}, border_color, fontsize=fontsize, **kwargs
        )
        t = 4 if labels_row is None else 3
        res = gtable_add_grob(res, elem, t=t, l=6, b=5, clip="off", name="annotation_legend")

    # Colour legend (col 5).
    if not is_na2(legend):
        elem = draw_legend(color, breaks, legend, fontsize=fontsize, **kwargs)
        t = 4 if labels_row is None else 3
        res = gtable_add_grob(res, elem, t=t, l=5, b=5, clip="off", name="legend")

    if filename is not None and not is_na2(filename):
        _save_to_file(res, filename, width, height)

    return res


def _save_to_file(
    gt: Any,
    filename: str,
    width: float | None,
    height: float | None,
) -> None:
    """Render *gt* to *filename*, mimicking R's pdf/png/jpeg/tiff/bmp branch.

    Width/height default to the gtable's intrinsic size (in inches).  Vector
    formats (pdf/svg/ps) are written via Cairo's vector surfaces; raster
    formats (png/jpeg/tiff/bmp) go through Cairo's image surface — jpeg/tiff/
    bmp additionally need PIL for the conversion from PNG.
    """
    from grid_py import convert_height, convert_width
    from grid_py._state import get_state
    from grid_py.renderer import CairoRenderer

    _, ext = os.path.splitext(str(filename))
    ext = ext.lower().lstrip(".")
    if ext not in {"pdf", "svg", "ps", "png", "jpeg", "jpg", "tiff", "bmp"}:
        raise ValueError("File type should be: pdf, png, bmp, jpg, tiff")

    # Mirror R: convertHeight(gtable_height(...), "inches") uses the active
    # device's dimensions to resolve `1npc` parts.  R's pdf/png default is
    # 7x7 inches; we install a temporary 7x7 measurement device so the
    # gtable's intrinsic size matches what R reports.
    state = get_state()
    measure_renderer = CairoRenderer(width=7.0, height=7.0, dpi=72.0,
                                     surface_type="image", bg="white")
    state.reset()
    state.init_device(measure_renderer)
    if height is None or (np.isscalar(height) and pd.isna(height)):
        height_in = float(np.atleast_1d(
            convert_height(gtable_height(gt), "in", valueOnly=True)
        ).ravel()[0])
    else:
        height_in = float(height)
    if width is None or (np.isscalar(width) and pd.isna(width)):
        width_in = float(np.atleast_1d(
            convert_width(gtable_width(gt), "in", valueOnly=True)
        ).ravel()[0])
    else:
        width_in = float(width)

    # R uses res=300 for raster formats; pdf has no DPI concept.
    if ext in {"pdf", "svg", "ps"}:
        renderer = CairoRenderer(
            width=width_in, height=height_in, dpi=72.0,
            surface_type=ext, filename=str(filename), bg="white",
        )
        state.reset()
        state.init_device(renderer)
        grid_draw(gt)
        renderer.finish()
        return

    # Raster path: render via in-memory ImageSurface at 300 dpi, then write.
    renderer = CairoRenderer(
        width=width_in, height=height_in, dpi=300.0,
        surface_type="image", bg="white",
    )
    state.reset()
    state.init_device(renderer)
    grid_draw(gt)
    if ext == "png":
        renderer.write_to_png(str(filename))
        return

    # jpeg/jpg/tiff/bmp via Pillow.
    from io import BytesIO
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"Saving '{ext}' requires Pillow (pip install pillow)."
        ) from exc
    buf = BytesIO(renderer.to_png_bytes())
    img = Image.open(buf)
    pil_format = {"jpg": "JPEG", "jpeg": "JPEG", "tiff": "TIFF", "bmp": "BMP"}[ext]
    save_kwargs: dict[str, Any] = {}
    if pil_format == "JPEG":
        img = img.convert("RGB")
    elif pil_format == "TIFF":
        save_kwargs["compression"] = "tiff_lzw"  # R's tiff() default.
    img.save(str(filename), format=pil_format, **save_kwargs)
