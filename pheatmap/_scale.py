"""Row/column centring and scaling (``scale_rows`` / ``scale_mat``).

Ported from lines 557–569 of ``R/pheatmap.r``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["scale_rows", "scale_mat"]


def scale_rows(x: np.ndarray) -> np.ndarray:
    """Centre each row at zero and scale by its sample standard deviation.

    Parameters
    ----------
    x
        Numeric 2-D array; NaNs are ignored in the mean/sd computation.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``x`` with ``(x - rowmean) / rowsd``.
    """
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    return (x - m) / s


def scale_mat(mat: np.ndarray, scale: str) -> np.ndarray:
    """Scale a matrix by row, column, or neither.

    Parameters
    ----------
    mat
        Numeric matrix.
    scale
        One of ``"none"``, ``"row"``, or ``"column"``.

    Returns
    -------
    np.ndarray
        Scaled matrix.

    Raises
    ------
    ValueError
        If ``scale`` is not one of the three accepted values.
    """
    if scale not in ("none", "row", "column"):
        raise ValueError(
            "scale argument should take values: 'none', 'row' or 'column'"
        )
    if scale == "none":
        return np.asarray(mat, dtype=float)
    if scale == "row":
        return scale_rows(mat)
    return scale_rows(np.asarray(mat, dtype=float).T).T
