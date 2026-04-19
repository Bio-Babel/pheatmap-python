"""Colour utilities: map numeric matrices to colour matrices, convert
annotation data frames to colour matrices, auto-assign annotation palettes.

Ported from ``scale_vec_colours``, ``scale_colours``, ``convert_annotations``,
``generate_annotation_colours`` in ``R/pheatmap.r``.
"""

from __future__ import annotations

import contextlib
import random
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scales import colour_ramp, dscale, hue_pal, pal_brewer

__all__ = [
    "scale_vec_colours",
    "scale_colours",
    "convert_annotations",
    "generate_annotation_colours",
    "colorRampPalette",  # noqa: N802  (preserve R-compatible name for users)
    "colour_ramp_palette",
    "r_style_cut",
]


def colour_ramp_palette(colors: Sequence[str], bias: float = 1.0) -> Any:
    """R's ``grDevices::colorRampPalette`` analogue.

    Parameters
    ----------
    colors
        Two-or-more hex/CSS colour names that anchor the ramp.
    bias
        Currently unused; present for signature parity with R.

    Returns
    -------
    callable
        A callable that, given an integer ``n``, returns ``n`` interpolated
        hex strings.
    """
    ramp = colour_ramp(list(colors))

    def _f(n: int) -> list[str]:
        if n <= 0:
            return []
        if n == 1:
            return ramp([0.5])
        xs = np.linspace(0.0, 1.0, n)
        return list(ramp(xs))

    return _f


# Alias so users coming from R find the camelCase name.
colorRampPalette = colour_ramp_palette  # noqa: N816


def r_style_cut(x: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Replicate R's ``cut(x, breaks, include.lowest=TRUE)`` index-return.

    Returns a 0-based integer index for each value (or ``-1`` for NaN / out of
    range).  ``-1`` is then used by :func:`scale_vec_colours` to fill the NA
    colour.
    """
    x = np.asarray(x, dtype=float)
    breaks = np.asarray(breaks, dtype=float)
    finite_breaks = breaks[np.isfinite(breaks)]
    lo = finite_breaks.min() if finite_breaks.size else -np.inf
    hi = finite_breaks.max() if finite_breaks.size else np.inf
    idx = np.searchsorted(breaks, x, side="right") - 1
    idx = np.where(np.isnan(x), -1, idx)
    idx = np.clip(idx, -1, len(breaks) - 2)
    # Values equal to the lowest finite break in R map to bin 0 (because
    # include.lowest=TRUE).  With -Inf sentinel this is handled already;
    # double-check for the no-sentinel path:
    eq_lo = (x == lo) & (idx == -1) & ~np.isnan(x)
    idx = np.where(eq_lo, 0, idx)
    # Values equal to hi go in the last bin.
    eq_hi = (x == hi) & (idx >= len(breaks) - 1) & ~np.isnan(x)
    idx = np.where(eq_hi, len(breaks) - 2, idx)
    return idx


def scale_vec_colours(
    x: np.ndarray,
    col: Sequence[str],
    breaks: np.ndarray,
    na_col: str,
) -> np.ndarray:
    """Vectorised value → colour lookup using ``cut(x, breaks)``."""
    col = np.asarray(col, dtype=object)
    idx = r_style_cut(x, breaks)
    out = np.empty_like(idx, dtype=object)
    mask_valid = idx >= 0
    out[mask_valid] = col[idx[mask_valid]]
    out[~mask_valid] = na_col
    return out


def scale_colours(
    mat: np.ndarray,
    col: Sequence[str],
    breaks: np.ndarray,
    na_col: str,
) -> np.ndarray:
    """Map a numeric matrix to a matrix of hex colour strings."""
    mat = np.asarray(mat, dtype=float)
    flat = scale_vec_colours(mat.ravel(order="F"), col, breaks, na_col)
    return flat.reshape(mat.shape, order="F")


def convert_annotations(
    annotation: pd.DataFrame,
    annotation_colors: Mapping[str, Any],
) -> np.ndarray:
    """Convert an annotation DataFrame to a matrix of hex colours.

    Parameters
    ----------
    annotation
        DataFrame whose columns are either categorical (``object`` /
        ``CategoricalDtype``) or numeric.
    annotation_colors
        Mapping from column name to either a list of hex strings (continuous
        ramp) or a mapping of level → colour (categorical).

    Returns
    -------
    np.ndarray
        Object array of the same shape as ``annotation`` containing hex
        colours.
    """
    out = np.full(annotation.shape, "", dtype=object)
    for j, col in enumerate(annotation.columns):
        a = annotation[col]
        b = annotation_colors[col]
        if isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object:
            a_str = a.astype(str).where(~a.isna(), other=None)
            unknown = {
                v for v in a_str.dropna().unique()
                if v not in b
            }
            if unknown:
                raise ValueError(
                    f"Factor levels on variable {col} do not match with "
                    "annotation_colors"
                )
            for i, v in enumerate(a_str):
                if v is None:
                    out[i, j] = None
                else:
                    out[i, j] = b[v]
        else:
            finite = np.isfinite(a.values.astype(float))
            idx = np.full(a.shape, -1, dtype=int)
            if finite.any():
                xmin, xmax = np.nanmin(a), np.nanmax(a)
                if xmin == xmax:
                    idx[finite] = 0
                else:
                    bins = np.linspace(xmin, xmax, 101)
                    bi = np.searchsorted(bins, a.values, side="right") - 1
                    bi = np.clip(bi, 0, 99)
                    idx[finite] = bi[finite]
            ramp = colour_ramp_palette(list(b))(100)
            for i, bi in enumerate(idx):
                out[i, j] = ramp[bi] if bi >= 0 else None
    return out


def _r_sample(rng: random.Random, n: int, k: int) -> list[int]:
    """Emulate R's ``sample(1:n, k)`` enough for palette index selection.

    We do not reproduce R's exact RNG stream (that would require a bit-exact
    Mersenne Twister port); instead we use Python's :mod:`random` seeded to
    the same seed.  The tutorial's annotation colour check is locked to our
    own fixture rather than to R's raw byte stream.
    """
    return rng.sample(range(n), k)


def generate_annotation_colours(
    annotation: "pd.DataFrame | Mapping[str, pd.Series]",
    annotation_colors: Any,
    drop: bool = True,
) -> Dict[str, Any]:
    """Fill in missing entries of ``annotation_colors``.

    For categorical columns a subset of a hue-based palette is assigned and
    named with the factor levels.  For continuous columns a 4-step sequential
    Brewer palette is assigned, cycling through palette numbers.

    Parameters
    ----------
    annotation
        A DataFrame or a mapping ``{name -> Series}`` (R's concatenation of
        ``annotation_col`` and ``annotation_row``).
    annotation_colors
        Existing colour mapping.  May be ``None`` / ``NaN`` for "empty".
    drop
        If ``True``, drop unused factor levels.

    Returns
    -------
    dict
        Mapping ``{column name -> palette}`` for every column in
        ``annotation``.
    """
    if annotation_colors is None or (
        not isinstance(annotation_colors, dict) and pd.isna(annotation_colors)
    ):
        annotation_colors = {}
    else:
        annotation_colors = dict(annotation_colors)

    if isinstance(annotation, pd.DataFrame):
        items = [(c, annotation[c]) for c in annotation.columns]
    else:
        items = list(annotation.items())

    # Count slots needed in the qualitative palette.
    count = 0
    for _, a in items:
        a = a.dropna()
        if isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object:
            if isinstance(a.dtype, pd.CategoricalDtype) and not drop:
                count += len(a.cat.categories)
            else:
                count += a.nunique()

    if count > 0:
        factor_colors = list(dscale(list(range(1, count + 1)), hue_pal(l=75)))
    else:
        factor_colors = []

    rng = random.Random(3453)
    with contextlib.suppress(Exception):
        np_state = np.random.get_state()

    cont_counter = 2
    for name, a in items:
        if name in annotation_colors:
            continue
        a = a.dropna()
        if isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object:
            if isinstance(a.dtype, pd.CategoricalDtype) and not drop:
                levels = list(a.cat.categories)
            else:
                levels = list(pd.unique(a.astype(str)))
                # Use sorted order for unnamed character columns, matching R's
                # ``levels(as.factor(...))`` ordering.
                levels = sorted(levels)
            n = len(levels)
            ind = _r_sample(rng, len(factor_colors), n)
            picked = [factor_colors[i] for i in ind]
            annotation_colors[name] = dict(zip(levels, picked))
            factor_colors = [c for i, c in enumerate(factor_colors) if i not in ind]
        else:
            palette_name_seq = [
                "Blues", "Greens", "Oranges", "Reds", "Purples",
                "Greys", "BuGn", "BuPu", "GnBu", "OrRd",
                "PuBu", "PuBuGn", "PuRd", "RdPu", "YlGn",
                "YlGnBu", "YlOrBr", "YlOrRd",
            ]
            name_idx = (cont_counter - 1) % len(palette_name_seq)
            palette_name = palette_name_seq[name_idx]
            pal5 = pal_brewer(type="seq", palette=palette_name)(5)
            annotation_colors[name] = list(pal5[:4])
            cont_counter += 1

    with contextlib.suppress(Exception):
        np.random.set_state(np_state)  # type: ignore[name-defined]

    return annotation_colors
