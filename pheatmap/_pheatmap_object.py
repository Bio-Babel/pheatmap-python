"""Python-side replacement for R's ``pheatmap`` S3 class plus its
``grid.draw`` / ``print`` methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grid_py import grid_draw as _grid_draw_impl
from grid_py import grid_newpage

__all__ = ["PHeatmap", "grid_draw"]


@dataclass
class PHeatmap:
    """Container returned by :func:`pheatmap.pheatmap`.

    Attributes
    ----------
    tree_row
        The row :class:`pheatmap._cluster.HClust` object, or ``None`` when
        rows are not clustered.
    tree_col
        Same for columns.
    kmeans
        K-means summary dict (``{"centers", "cluster", "sizes"}``) or ``None``
        if k-means was not requested.
    gtable
        The composed :class:`gtable_py.Gtable`.
    """

    tree_row: Any
    tree_col: Any
    kmeans: Any
    gtable: Any

    def draw(self) -> None:
        """Render the heatmap to the current grid device."""
        _grid_draw_impl(self.gtable)

    def __repr__(self) -> str:
        return (
            f"PHeatmap(gtable={self.gtable!r}, "
            f"tree_row={'clustered' if self.tree_row is not None else None}, "
            f"tree_col={'clustered' if self.tree_col is not None else None}, "
            f"kmeans={'yes' if self.kmeans is not None else None})"
        )

    def _ipython_display_(self) -> None:  # pragma: no cover - UI only
        try:
            grid_newpage()
        except Exception:
            pass
        _grid_draw_impl(self.gtable)


def grid_draw(x: Any, recording: bool = True) -> None:
    """Module-level ``grid.draw`` dispatch for :class:`PHeatmap` and generic grobs."""
    if isinstance(x, PHeatmap):
        _grid_draw_impl(x.gtable, recording=recording)
    else:
        _grid_draw_impl(x, recording=recording)
