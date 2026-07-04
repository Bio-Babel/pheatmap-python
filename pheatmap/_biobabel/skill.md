---
name: use-pheatmap
description: Draw a clustered, annotated heatmap from a numeric matrix or DataFrame, matching R's pheatmap() output, without matplotlib.
---

# pheatmap

Python port of Raivo Kolde's R **pheatmap** package (v1.0.13, commit `ffd0f8c`).
Renders through `grid_py`'s Cairo backend (`grid`/`gtable_py`/`scales` ports),
not matplotlib. One call — `pheatmap(mat, ...)` — coerces the input, optionally
scales/clusters/k-means-aggregates rows or columns, colours the matrix, and
composes+draws a `gtable`. There is no multi-step pipeline to build: nearly
every documented usage is a single `pheatmap()` call with different keyword
arguments (scale, clustering, annotations, gaps, display numbers, colours,
file export). The many other public symbols (`cluster_mat`, `scale_mat`,
`generate_breaks`, `draw_*`, ...) are the internals `pheatmap()` calls,
re-exported for advanced users who want to inspect or reuse an intermediate
step, not a separate workflow.

## When to use

- You have a numeric matrix/DataFrame and want a clustered heatmap with
  dendrograms, annotation tracks, a colour legend, and/or on-cell numbers.
- You need output saved to pdf/png/svg/ps/jpeg/tiff/bmp, or a raster-embedded
  body to keep file size bounded on very large matrices (`use_raster=True`).
- You are porting R code that calls `pheatmap::pheatmap(...)` and need
  matching behaviour.

## When not to use

- You need the result as a matplotlib `Figure`/`Axes` — pheatmap draws via
  grid_py's Cairo backend and cannot be composed into a matplotlib figure.
- You need an interactive or zoomable heatmap — pheatmap only produces a
  static render.

## Entry points

- `pheatmap.pheatmap(mat, ...) -> PHeatmap` — the one function almost every
  use case needs.
- `PHeatmap.draw()` / `pheatmap.grid_draw(result)` — re-render a result
  returned with `silent=True`.
- `pheatmap.cluster_mat`, `pheatmap.scale_mat`, `pheatmap.generate_breaks` —
  the internal steps, exposed for advanced inspection/reuse.

## Quick reference

```python
import numpy as np
import pandas as pd
from pheatmap import pheatmap

mat = pd.DataFrame(
    np.random.default_rng(1).standard_normal((20, 10)),
    index=[f"Gene{i+1}" for i in range(20)],
    columns=[f"Test{i+1}" for i in range(10)],
)
pheatmap(mat, scale="row", clustering_distance_rows="correlation")
```

For more: `biobabel.describe_package(import_name="pheatmap")`.
