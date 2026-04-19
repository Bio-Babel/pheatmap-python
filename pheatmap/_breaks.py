"""Linear break sequences used for colour mapping.

Ported from ``generate_breaks`` in ``R/pheatmap.r``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["generate_breaks"]


def generate_breaks(
    x: np.ndarray,
    n: int,
    center: bool = False,
) -> np.ndarray:
    """Return ``n + 1`` break points that span ``x``.

    Parameters
    ----------
    x
        Array-like numeric data.
    n
        Number of colour bins.  The returned vector has ``n + 1`` points.
    center
        If ``True``, breaks are symmetric around zero with magnitude equal to
        ``max(abs(min(x)), abs(max(x)))``.  Otherwise they run from
        ``min(x)`` to ``max(x)``.

    Returns
    -------
    np.ndarray
        Strictly increasing length-``n + 1`` vector.
    """
    x = np.asarray(x, dtype=float)
    if center:
        m = np.nanmax(np.abs([np.nanmin(x), np.nanmax(x)]))
        return np.linspace(-m, m, n + 1)
    return np.linspace(np.nanmin(x), np.nanmax(x), n + 1)
