"""Grob builders used by :func:`pheatmap._motor.heatmap_motor`.

Ported from lines 146–375 of ``R/pheatmap.r``.  Each function returns a
``grid_py`` grob (``GTree``, ``text_grob``, ``rect_grob``, …).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from grid_py import (
    GList,
    GTree,
    Gpar,
    Unit,
    grob_tree,
    grid_pretty,
    polyline_grob,
    rect_grob,
    string_height,
    text_grob,
    unit_c,
    unit_rep,
)
from grid_py._units import unit_summary_min

from ._colours import colour_ramp_palette, convert_annotations
from ._layout import find_coordinates
from ._utils import is_na2

__all__ = [
    "draw_dendrogram",
    "draw_matrix",
    "draw_colnames",
    "draw_rownames",
    "draw_legend",
    "draw_annotations",
    "draw_annotation_names",
    "draw_annotation_legend",
    "draw_main",
]


def _u(v: float, units: str = "bigpts") -> Unit:
    return Unit(v, units)


def _vals(u: Unit) -> np.ndarray:
    """Return the underlying numeric values of a :class:`Unit` as a 1-D array.

    ``grid_py.Unit`` exposes ``.values`` rather than ``.value``; this helper
    centralises the accessor so call-sites don't care about the attribute name.
    """
    return np.atleast_1d(np.asarray(u.values, dtype=float))


def _first_units(u: Unit) -> str:
    """Return the first unit string of a :class:`Unit` (or ``"npc"`` by default)."""
    lst = getattr(u, "units_list", None)
    if lst:
        return str(lst[0])
    return "npc"


def draw_dendrogram(hc: Any, gaps: Sequence[int] | None, horizontal: bool = True):
    """Render a dendrogram with optional gaps."""
    height = np.asarray(hc.height, dtype=float)
    h = height / height.max() / 1.05
    m = np.asarray(hc.merge, dtype=int).copy()
    o = np.asarray(hc.order, dtype=int)
    n = len(o)

    # Use 1-based internally (matches R's merge semantics).
    m = m.astype(float)
    positive = m > 0
    negative = m < 0
    m[positive] = n + m[positive]
    m[negative] = np.abs(m[negative])
    m = m.astype(int)

    # dist[:, 0] is x-coord (in cells), dist[:, 1] is height.
    dist = np.zeros((2 * n - 1, 2), dtype=float)
    # match(1:n, o) in R is positions of 1..n inside o; using 1-based.
    inv = np.argsort(o) + 1
    dist[:n, 0] = 1.0 / n / 2.0 + (1.0 / n) * (inv - 1)

    for i in range(n - 1):
        a, b = m[i, 0], m[i, 1]
        # Convert R 1-based merge reference to 0-based Python row in `dist`.
        # Observation k (1..n) is row k-1; cluster r is row n + r - 1.
        ra = a - 1 if a <= n else n + (a - n) - 1
        rb = b - 1 if b <= n else n + (b - n) - 1
        # Actually our R-to-scipy conversion used "merge[i,0] = -(a+1) if a<n else (a-n+1)",
        # meaning for original obs idx `a` (0-based in scipy), merge entry = -(a+1),
        # then in this draw routine we applied abs() and got `a+1`.  So for
        # original obs k (1-based), dist row = k - 1.  For cluster id c (>=1),
        # merge entry here equals n + c, and dist row = n + c - 1.
        dist[n + i, 0] = (dist[ra, 0] + dist[rb, 0]) / 2.0
        dist[n + i, 1] = h[i]

    xs = np.zeros(4 * (n - 1), dtype=float)
    ys = np.zeros(4 * (n - 1), dtype=float)
    ids = np.repeat(np.arange(1, n), 4)

    for i in range(n - 1):
        a, b = m[i, 0], m[i, 1]
        ra = a - 1 if a <= n else n + (a - n) - 1
        rb = b - 1 if b <= n else n + (b - n) - 1
        x1, x2 = dist[ra, 0], dist[rb, 0]
        y1, y2 = dist[ra, 1], dist[rb, 1]
        y = h[i]
        k = i * 4
        xs[k : k + 4] = [x1, x1, x2, x2]
        ys[k : k + 4] = [y1, y, y, y2]

    coord = find_coordinates(n, gaps, xs * n)
    x = coord["coord"]
    y = Unit(ys, "npc")

    if not horizontal:
        a = x
        x = Unit(1.0, "npc") - y
        y = Unit(1.0, "npc") - a

    return polyline_grob(x=x, y=y, id=list(ids))


def draw_matrix(
    matrix: np.ndarray,
    border_color: str | None,
    gaps_rows: Sequence[int] | None,
    gaps_cols: Sequence[int] | None,
    fmat: np.ndarray,
    fontsize_number: float,
    number_color: str,
    draw_numbers: bool,
) -> GTree:
    """Render the coloured matrix grid and optional cell-value text."""
    n, m = matrix.shape
    coord_x = find_coordinates(m, gaps_cols)
    coord_y = find_coordinates(n, gaps_rows)

    x_half = coord_x["size"] * 0.5
    y_half = coord_y["size"] * 0.5
    x = coord_x["coord"] - x_half
    y = Unit(1.0, "npc") - (coord_y["coord"] - y_half)

    # expand.grid(y = y, x = x) → y varies fastest, x is replicated per y.
    # Use unit_rep to preserve compound-unit structure when gaps are present.
    x_g = unit_rep(x, each=n)
    y_g = unit_rep(y, times=m)

    # Flatten colours in column-major order (R's `as.vector(mat)`).
    fill = np.asarray(matrix, dtype=object).ravel(order="F")

    rect = rect_grob(
        x=x_g,
        y=y_g,
        width=coord_x["size"],
        height=coord_y["size"],
        gp=Gpar(fill=list(fill), col=border_color),
    )

    if draw_numbers:
        labels = np.asarray(fmat, dtype=object).ravel(order="F")
        text = text_grob(
            list(labels),
            x=x_g,
            y=y_g,
            gp=Gpar(col=number_color, fontsize=fontsize_number),
        )
        return grob_tree(rect, text)
    return grob_tree(rect)


def draw_colnames(
    coln: Sequence[str],
    gaps: Sequence[int] | None,
    vjust_col: float,
    hjust_col: float,
    angle_col: float,
    **kwargs: Any,
):
    coord = find_coordinates(len(coln), gaps)
    x = coord["coord"] - coord["size"] * 0.5
    return text_grob(
        list(coln),
        x=x,
        y=Unit(1.0, "npc") - _u(3.0),
        vjust=vjust_col,
        hjust=hjust_col,
        rot=angle_col,
        gp=Gpar(**kwargs),
    )


def draw_rownames(
    rown: Sequence[str],
    gaps: Sequence[int] | None,
    **kwargs: Any,
):
    coord = find_coordinates(len(rown), gaps)
    y = Unit(1.0, "npc") - (coord["coord"] - coord["size"] * 0.5)
    return text_grob(
        list(rown),
        x=_u(3.0),
        y=y,
        vjust=0.5,
        hjust=0.0,
        gp=Gpar(**kwargs),
    )


def draw_legend(
    color: Sequence[str],
    breaks: np.ndarray,
    legend: Mapping[str, float] | Sequence[float],
    **kwargs: Any,
) -> GTree:
    color = np.asarray(color)
    breaks = np.asarray(breaks, dtype=float)
    # R does `color = color[!is.infinite(breaks)]` with recycling, the net
    # effect of which (after pheatmap()'s -Inf/+Inf augmentation) is to drop
    # the sentinel-duplicated colours at each augmented end so that the
    # remaining colours line up with the finite breaks.
    if np.isneginf(breaks[0]) and len(color) > 0:
        color = color[1:]
    if np.isposinf(breaks[-1]) and len(color) > 0:
        color = color[:-1]
    breaks = breaks[np.isfinite(breaks)]

    if isinstance(legend, dict):
        labels = list(legend.keys())
        vals = np.asarray([float(v) for v in legend.values()], dtype=float)
    else:
        arr = np.asarray(list(legend), dtype=float)
        vals = arr
        labels = [str(v) for v in arr]

    # R: height = min(unit(1, "npc"), unit(150, "bigpts"))
    height = unit_summary_min(Unit(1.0, "npc"), _u(150.0))

    span = breaks.max() - breaks.min()
    if span == 0:
        span = 1.0
    legend_pos = (vals - breaks.min()) / span
    offset = Unit(1.0, "npc") - height
    legend_pos_unit = unit_c(*[height * float(p) for p in legend_pos]) + offset

    bnorm = (breaks - breaks.min()) / span
    breaks_unit = unit_c(*[height * float(p) for p in bnorm]) + offset

    # R: h = breaks[-1] - breaks[-length(breaks)] — keep this as a unit
    # subtraction so the heights stay correct when `height` is the compound
    # min(1npc, 150bigpts) rather than a single bigpts scalar.
    h_unit = breaks_unit[1:] - breaks_unit[:-1]
    y_bot = breaks_unit[:-1]

    rect = rect_grob(
        x=Unit(0.0, "npc"),
        y=y_bot,
        width=_u(10.0),
        height=h_unit,
        hjust=0.0,
        vjust=0.0,
        gp=Gpar(fill=list(color), col="#FFFFFF00"),
    )

    text = text_grob(
        labels,
        x=_u(14.0),
        y=legend_pos_unit,
        hjust=0.0,
        gp=Gpar(**kwargs),
    )

    return grob_tree(rect, text)


def draw_annotations(
    converted_annotations: np.ndarray,
    border_color: str | None,
    gaps: Sequence[int] | None,
    fontsize: float,
    horizontal: bool,
):
    m, n = converted_annotations.shape  # rows = observations, cols = tracks
    coord_x = find_coordinates(m, gaps)
    x = coord_x["coord"] - coord_x["size"] * 0.5

    y_vals = np.cumsum(np.full(n, fontsize)) + np.cumsum(np.full(n, 2.0)) - fontsize / 2 + 1
    y_unit = Unit(y_vals, "bigpts")

    if horizontal:
        # expand.grid(x=x, y=y) → x varies fastest, y is replicated per x.
        xg = unit_rep(x, times=n)
        yg = unit_rep(y_unit, each=m)
        fill = converted_annotations.ravel(order="F")
        return rect_grob(
            x=xg,
            y=yg,
            width=coord_x["size"],
            height=_u(fontsize),
            gp=Gpar(fill=list(fill), col=border_color),
        )
    # vertical
    # a = x; x = 1npc - y; y = 1npc - a
    # expand.grid(y = y, x = x) → y varies fastest
    new_x = Unit(1.0, "npc") - y_unit  # length n
    new_y = Unit(1.0, "npc") - x       # length m

    yg = unit_rep(new_y, times=n)
    xg = unit_rep(new_x, each=m)

    # Fill order matches expand.grid(y=new_y[len=m], x=new_x[len=n]): y varies
    # fastest, so element k = (row=k%m, track=k//m). That is column-major over
    # the (m, n) annotation matrix.
    fill = converted_annotations.ravel(order="F")

    return rect_grob(
        x=xg,
        y=yg,
        width=_u(fontsize),
        height=coord_x["size"],
        gp=Gpar(fill=list(fill), col=border_color),
    )


def draw_annotation_names(
    annotations: pd.DataFrame,
    fontsize: float,
    horizontal: bool,
    hjust_col: float = 0.0,
    vjust_col: float = 0.5,
    angle_col: float = 0.0,
):
    n = annotations.shape[1]
    x = _u(3.0)
    y_vals = (
        np.cumsum(np.full(n, fontsize))
        + np.cumsum(np.full(n, 2.0))
        - fontsize / 2
        + 1
    )
    y = Unit(y_vals, "bigpts")

    # R uses fontface=2 for annotation names (= bold).
    if horizontal:
        return text_grob(
            list(annotations.columns),
            x=x,
            y=y,
            hjust=0.0,
            gp=Gpar(fontsize=fontsize, fontface="bold"),
        )
    new_x = Unit(1.0, "npc") - y
    new_y = Unit(1.0, "npc") - x
    return text_grob(
        list(annotations.columns),
        x=new_x,
        y=new_y,
        vjust=vjust_col,
        hjust=hjust_col,
        rot=angle_col,
        gp=Gpar(fontsize=fontsize, fontface="bold"),
    )


def draw_annotation_legend(
    annotation: Mapping[str, pd.Series],
    annotation_colors: Mapping[str, Any],
    border_color: str | None,
    **kwargs: Any,
) -> GTree:
    """Compose the annotation legend as a stack of name + swatches."""
    y = Unit(1.0, "npc")
    text_h = string_height("FGH")

    grobs: list = []

    for name, series in annotation.items():
        gp_kwargs = {"fontface": "bold", **kwargs}
        grobs.append(
            text_grob(
                name,
                x=Unit(0.0, "npc"),
                y=y,
                vjust=1.0,
                hjust=0.0,
                gp=Gpar(**gp_kwargs),
            )
        )
        y = y - text_h * 1.5
        colours = annotation_colors[name]
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
            if isinstance(colours, dict):
                names = list(colours.keys())
                values = list(colours.values())
            else:
                names = list(map(str, colours))
                values = list(colours)
            num = len(names)
            yy = y - unit_c(*[text_h * float(i * 2.0) for i in range(num)])
            grobs.append(
                rect_grob(
                    x=Unit(0.0, "npc"),
                    y=yy,
                    hjust=0.0,
                    vjust=1.0,
                    height=text_h * 2.0,
                    width=text_h * 2.0,
                    gp=Gpar(col=border_color, fill=values),
                )
            )
            grobs.append(
                text_grob(
                    names,
                    x=text_h * 2.4,
                    y=yy - text_h,
                    hjust=0.0,
                    vjust=0.5,
                    gp=Gpar(**kwargs),
                )
            )
            y = y - text_h * (num * 2.0)
        else:
            # Continuous: 4 stacked gradient blocks + surrounding border.
            ramp4 = colour_ramp_palette(list(colours))(4)
            base = y - text_h * 8.0
            fracs = np.arange(1, 5) * 0.25
            yy = base + unit_c(*[text_h * float(f * 8.0) for f in fracs])
            h = text_h * (8.0 * 0.25)
            grobs.append(
                rect_grob(
                    x=Unit(0.0, "npc"),
                    y=yy,
                    hjust=0.0,
                    vjust=1.0,
                    height=h,
                    width=text_h * 2.0,
                    gp=Gpar(col=None, fill=list(ramp4)),
                )
            )
            grobs.append(
                rect_grob(
                    x=Unit(0.0, "npc"),
                    y=y,
                    hjust=0.0,
                    vjust=1.0,
                    height=text_h * 8.0,
                    width=text_h * 2.0,
                    gp=Gpar(col=border_color, fill=None),
                )
            )
            vmin, vmax = float(np.nanmin(series)), float(np.nanmax(series))
            rng = grid_pretty((vmin, vmax))
            txt = [str(v) for v in rng[::-1][[0, -1]] if True][:2]  # rev(range(pretty))
            # Correct: reverse and take (first, last); just use rng[0] and rng[-1]
            txt = [str(rng[-1]), str(rng[0])]
            yy2 = y - unit_c(*[text_h * float(v) for v in (1.0, 7.0)])
            grobs.append(
                text_grob(
                    txt,
                    x=text_h * 2.4,
                    y=yy2,
                    hjust=0.0,
                    vjust=0.5,
                    gp=Gpar(**kwargs),
                )
            )
            y = y - text_h * 8.0
        y = y - text_h * 1.5

    return grob_tree(*grobs)


def draw_main(text: str, **kwargs: Any):
    # R uses gpar(fontface="bold", ...) which lets the "..." override the
    # bold default if the caller asks for it; we replicate that by letting
    # the caller's fontface win.
    gp_kwargs = {"fontface": "bold", **kwargs}
    return text_grob(text, gp=Gpar(**gp_kwargs))
