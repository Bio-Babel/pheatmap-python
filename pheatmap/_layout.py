"""Grid layout computations (``find_coordinates`` / ``lo``).

Ported from lines 1–144 of ``R/pheatmap.r``.  We build widths and heights
from ``grid_py`` units and return a 5-row × 6-column :class:`gtable_py.Gtable`
skeleton that the drawing routines populate.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from grid_py import (
    Gpar,
    Unit,
    convert_height,
    convert_width,
    string_height,
    string_width,
    unit_c,
)
from gtable_py import Gtable

from ._utils import is_na2

__all__ = ["find_coordinates", "lo"]


def _u(value: float, units: str = "bigpts") -> Unit:
    """Small convenience wrapper: `Unit(value, units)`."""
    return Unit(value, units)


def _npc() -> Unit:
    return Unit(1.0, "npc")


def _as_bigpts(u: Unit) -> float:
    """Convert a scalar :class:`Unit` to big-points as a plain ``float``."""
    arr = np.atleast_1d(np.asarray(convert_width(u, "bigpts", valueOnly=True)))
    return float(arr.ravel()[0])


def find_coordinates(
    n: int,
    gaps: Sequence[int] | None,
    m: Sequence[int] | None = None,
) -> dict[str, Unit]:
    """Compute per-cell coordinates and cell size along one axis.

    Parameters
    ----------
    n
        Number of cells along the axis.
    gaps
        1-based gap positions (number of cells *before* each gap).  May be
        ``None`` or an empty sequence.
    m
        Cell indices to compute coordinates for; defaults to ``1..n``.

    Returns
    -------
    dict
        ``{"coord": Unit-vector, "size": Unit-scalar}``.
    """
    gaps = [] if gaps is None else list(gaps)
    if m is None:
        m = np.arange(1, n + 1)
    else:
        m = np.asarray(m, dtype=float)

    if len(gaps) == 0:
        coord = Unit(np.asarray(m) / n, "npc")
        size = Unit(1.0 / n, "npc")
        return {"coord": coord, "size": size}

    if max(gaps) > n:
        raise ValueError("Gaps do not match with matrix size")

    size_coef = 1.0 / n
    # size = (1/n) * (1npc - len(gaps)*4bigpts)
    size = (_npc() - _u(len(gaps) * 4.0)) * size_coef

    # gaps2[k] = sum_i 1{m[k] > gaps[i]}
    gaps_arr = np.asarray(gaps, dtype=float)
    gaps2 = np.array([(mi > gaps_arr).sum() for mi in m], dtype=float)

    # coord = m * size + gaps2 * 4bigpts; `size` is a scalar Unit, `m` and
    # `gaps2` are plain arrays, so fan out with unit_c to stay element-wise.
    from grid_py import unit_c as _unit_c  # local import to avoid cycle
    m_arr = np.asarray(m, dtype=float)
    scaled = _unit_c(*[size * float(mi) for mi in m_arr])
    offsets = Unit(gaps2 * 4.0, "bigpts")
    coord = scaled + offsets
    return {"coord": coord, "size": size}


def _longest(strings: Sequence[str], cex_widths: np.ndarray) -> int:
    """Index of the longest rendered string (by approximate width)."""
    return int(np.argmax(cex_widths))


def lo(
    rown: Sequence[str] | None,
    coln: Sequence[str] | None,
    nrow: int,
    ncol: int,
    cellheight: float | None,
    cellwidth: float | None,
    treeheight_col: float,
    treeheight_row: float,
    legend: Any,
    annotation_row: pd.DataFrame | None,
    annotation_col: pd.DataFrame | None,
    annotation_colors: Any,
    annotation_legend: bool,
    annotation_names_row: bool,
    annotation_names_col: bool,
    main: str | None,
    fontsize: float,
    fontsize_row: float,
    fontsize_col: float,
    angle_col: float,
    gaps_row: Sequence[int] | None,
    gaps_col: Sequence[int] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute the enclosing :class:`gtable_py.Gtable` and minimum cell
    dimension used to decide if cell borders get drawn.

    Returns
    -------
    dict
        ``{"gt": Gtable, "mindim": float}``.  ``mindim`` is the smaller of
        the cell width and cell height in big-points.
    """
    # ----- Column names height -----
    if (coln is not None and len(coln) > 0) or (
        not is_na2(annotation_row) and annotation_names_row
    ):
        if coln is not None and len(coln) > 0:
            t = list(coln)
        else:
            t = [""]
        tw = np.array(
            [
                _as_bigpts(string_width(s)) / 72.0
                * (fontsize_col / fontsize)
                for s in t
            ]
        )
        if annotation_names_row and not is_na2(annotation_row):
            extra = list(annotation_row.columns)
            tw = np.concatenate(
                [tw, [_as_bigpts(string_width(s)) / 72.0 for s in extra]]
            )
            t = t + extra
        longest = _longest(t, tw)
        gp = Gpar(fontsize=(fontsize_col if longest < len(coln or []) else fontsize))
        # Height of a rotated text grob:
        h = _as_bigpts(string_height(t[longest]))
        # approximate rotation effect: for 0°, height; 90°/270°, width.
        if angle_col in (90.0, 270.0):
            coln_height = _u(
                _as_bigpts(string_width(t[longest]))
                + 10.0
            )
        elif angle_col in (45.0, 315.0):
            coln_height = _u(
                float(
                    convert_width(string_width(t[longest]), "bigpts", valueOnly=True)
                ) / np.sqrt(2)
                + 10.0
            )
        else:
            coln_height = _u(h + 10.0)
    else:
        coln_height = _u(5.0)

    # ----- Row names width -----
    if rown is not None and len(rown) > 0:
        t = list(rown)
        tw = np.array(
            [
                _as_bigpts(string_width(s)) / 72.0
                * (fontsize_row / fontsize)
                for s in t
            ]
        )
        if annotation_names_col and not is_na2(annotation_col):
            extra = list(annotation_col.columns)
            tw = np.concatenate(
                [tw, [_as_bigpts(string_width(s)) / 72.0 for s in extra]]
            )
            t = t + extra
        longest = _longest(t, tw)
        rown_width = _u(
            _as_bigpts(string_width(t[longest]))
            + 10.0
        )
    else:
        rown_width = _u(5.0)

    # ----- Legend width -----
    if not is_na2(legend):
        keys = list(legend.keys()) if isinstance(legend, dict) else [str(v) for v in legend]
        if keys:
            longest_key = keys[int(np.argmax([len(str(k)) for k in keys]))]
            legend_width = _u(
                12.0 + 1.1 * _as_bigpts(string_width(longest_key)) * 1.2
            )
            title_len = _u(1.1 * _as_bigpts(string_width("Scale")))
            legend_width = _u(max(_as_bigpts(legend_width), _as_bigpts(title_len)))
        else:
            legend_width = _u(0.0)
    else:
        legend_width = _u(0.0)

    # ----- Main title height -----
    if main is None or is_na2(main):
        main_height = Unit(0.0, "npc")
    else:
        h = _as_bigpts(string_height(main))
        main_height = _u(1.5 * h * 1.3)

    textheight = _u(fontsize)

    # ----- Column annotation block -----
    if not is_na2(annotation_col):
        ann_n_col = annotation_col.shape[1]
        annot_col_height = _u(ann_n_col * (fontsize + 2.0) + 2.0)
        t = list(annotation_col.astype(str).values.flatten()) + list(annotation_col.columns)
        longest_ann = t[int(np.argmax([len(str(x)) for x in t]))] if t else "X"
        annot_col_legend_width = _u(
            1.2 * _as_bigpts(string_width(longest_ann)) + 12.0
        )
        if not annotation_legend:
            annot_col_legend_width = _u(0.0)
    else:
        annot_col_height = _u(0.0)
        annot_col_legend_width = _u(0.0)

    # ----- Row annotation block -----
    if not is_na2(annotation_row):
        ann_n_row = annotation_row.shape[1]
        annot_row_width = _u(ann_n_row * (fontsize + 2.0) + 2.0)
        t = list(annotation_row.astype(str).values.flatten()) + list(annotation_row.columns)
        longest_ann = t[int(np.argmax([len(str(x)) for x in t]))] if t else "X"
        annot_row_legend_width = _u(
            1.2 * _as_bigpts(string_width(longest_ann)) + 12.0
        )
        if not annotation_legend:
            annot_row_legend_width = _u(0.0)
    else:
        annot_row_width = _u(0.0)
        annot_row_legend_width = _u(0.0)

    annot_legend_width = _u(max(_as_bigpts(annot_row_legend_width), _as_bigpts(annot_col_legend_width)))

    # ----- Tree heights / row tree widths -----
    treeheight_col_u = _u(treeheight_col + 5.0)
    treeheight_row_u = _u(treeheight_row + 5.0)

    # ----- Matrix block size -----
    gaps_row = list(gaps_row or [])
    gaps_col = list(gaps_col or [])

    if cellwidth is None or (np.isscalar(cellwidth) and pd.isna(cellwidth)):
        # mat_width = 1npc - rown_width - legend_width - treeheight_row - annot_row_width - annot_legend_width
        mat_width = (
            _npc()
            - rown_width
            - legend_width
            - treeheight_row_u
            - annot_row_width
            - annot_legend_width
        )
    else:
        mat_width = _u(cellwidth * ncol + len(gaps_col) * 4.0)

    if cellheight is None or (np.isscalar(cellheight) and pd.isna(cellheight)):
        mat_height = _npc() - main_height - coln_height - treeheight_col_u - annot_col_height
    else:
        mat_height = _u(cellheight * nrow + len(gaps_row) * 4.0)

    widths = unit_c(
        treeheight_row_u,
        annot_row_width,
        mat_width,
        rown_width,
        legend_width,
        annot_legend_width,
    )
    heights = unit_c(main_height, treeheight_col_u, annot_col_height, mat_height, coln_height)

    gt = Gtable(widths=widths, heights=heights)

    # cw = (mat_width - len(gaps_col)*4bigpts) / ncol  in bigpts
    cw = (_as_bigpts(mat_width) - len(gaps_col) * 4.0) / max(ncol, 1)
    ch = (_as_bigpts(mat_height) - len(gaps_row) * 4.0) / max(nrow, 1)

    mindim = float(min(cw, ch))

    return {"gt": gt, "mindim": mindim}
