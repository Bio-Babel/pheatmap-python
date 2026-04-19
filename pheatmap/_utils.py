"""Small helpers ported from ``pheatmap::is.na2`` and ``pheatmap::identity2``.

The R helpers treat ``NA`` specially: anything that is a list, has length
``> 1``, or has length zero is *not* "NA", while a scalar ``NA`` is.  We
reproduce the same contract in Python.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["is_na2", "identity2"]


def is_na2(x: Any) -> bool:
    """Return ``True`` iff ``x`` is a scalar NA-like value.

    Parameters
    ----------
    x
        Any Python object.

    Returns
    -------
    bool
        ``True`` when ``x`` is a scalar ``None`` / ``NaN`` / ``pd.NA``, ``False``
        otherwise (including all DataFrames, non-empty arrays, non-empty
        sequences, and non-NA scalars).
    """
    if isinstance(x, pd.DataFrame):
        return False
    if isinstance(x, (list, tuple, dict)):
        return len(x) == 0
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return bool(pd.isna(x.item()))
        return x.size == 0
    if isinstance(x, pd.Series):
        return x.empty
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def identity2(x: Any, *args: Any, **kwargs: Any) -> Any:
    """Return ``x`` unchanged; ignores additional arguments.

    Used as the default ``clustering_callback`` in :func:`pheatmap.pheatmap`.
    """
    return x
