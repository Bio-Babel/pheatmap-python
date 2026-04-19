# pheatmap_py

Python port of Raivo Kolde's R **pheatmap** package (v1.0.13).  Draws
pretty clustered heatmaps with control over cell size, clustering,
scaling, annotations, gaps, and legend formatting, returning a
[gtable_py](https://github.com/) object that can be composed with other
grid-based figures or saved to `.pdf`/`.png`/`.tiff`/`.jpeg`/`.bmp`.

## Installation

```bash
pip install -e .
```

The port depends on three sibling ports already available on the Python
side:

| R package | Python port    |
|-----------|---------------|
| `grid`    | `grid_py`     |
| `gtable`  | `gtable_py`   |
| `scales`  | `scales`      |

Together they cover the palette generation, coordinate system, and
rendering that the original `pheatmap` relied on.

## Quick start

```python
import numpy as np
import pandas as pd
from pheatmap import pheatmap

rng = np.random.default_rng(1)
mat = pd.DataFrame(
    rng.standard_normal((20, 10)),
    index=[f"Gene{i+1}" for i in range(20)],
    columns=[f"Test{i+1}" for i in range(10)],
)

pheatmap(mat)
```

The function returns a [`PHeatmap`][pheatmap.PHeatmap] with the row/column
dendrograms, the composed `gtable`, and (when `kmeans_k` is used) a
kmeans summary.

## Tutorial

A full port of the upstream R vignette is provided as a Jupyter notebook:

- [pheatmap tutorial](pheatmap_tutorial.ipynb)

## API

See the [API reference](api.md) for full signatures and docstrings.

## Parity with R

Semantic parity is locked by an R-side reference generator and a Python
validation script (`validation/tutorial_pheatmap_tutorial.py`).  Known
intentional deviations:

- The default k-means seed ``1245678`` is reused, but Python's k-means
  implementation (SciPy's ``kmeans2``) differs from R's Hartigan-Wong, so
  cluster identity numbers (though not the composition quality) may
  differ.
- `colorRampPalette`/`colour_ramp_palette` interpolate in RGB in both
  ports but the exact hex output may differ by one LSB in places due to
  rounding.
- The R-only colour names (e.g. `firebrick3`) are not understood; use hex
  equivalents or matplotlib-standard colour names.
