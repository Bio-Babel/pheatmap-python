"""Smoke test for pheatmap._biobabel.

Exercises the real end-to-end path: build a tiny synthetic matrix, draw a
clustered+annotated heatmap with pheatmap(), and check the returned PHeatmap
is populated. silent=True avoids opening a GUI/display; no network access,
no large data, nothing written outside a temp dir.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    from pheatmap import PHeatmap, pheatmap

    rng = np.random.default_rng(0)
    mat = pd.DataFrame(
        rng.standard_normal((6, 4)),
        index=[f"Gene{i + 1}" for i in range(6)],
        columns=[f"Sample{i + 1}" for i in range(4)],
    )
    annotation_col = pd.DataFrame(
        {"Group": ["A", "A", "B", "B"]}, index=mat.columns
    )

    res = pheatmap(mat, annotation_col=annotation_col, silent=True)
    assert isinstance(res, PHeatmap)
    assert res.gtable is not None
    assert res.tree_row is not None and res.tree_col is not None

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.pdf"
        pheatmap(mat, filename=str(out), silent=True)
        assert out.exists()

    print("pheatmap smoke test passed")


if __name__ == "__main__":
    main()
