# pheatmap-python

[![PyPI](https://img.shields.io/pypi/v/pheatmap-python)](https://pypi.org/project/pheatmap-python/)

Python version of **pheatmap** that runs on a faithful port of R's native **grid** graphics system instead of matplotlib.

- 🎨 **No matplotlib.** Renders through `grid_py`'s Cairo backend
  (PNG / PDF / SVG / PS / JPEG / TIFF / BMP).
- 📐 **Grammar of Graphics.** Built on `grid_py` + `gtable_py` — the
  same primitives ggplot2 stands on.
- 🧩 **Abstract layout.** Sizes stay as compound `Unit` expressions
  (`1npc - rown_width - legend_width - …`) and resolve at draw time.
- 🎯 **R is the gold standard.** Behaviour almost like the `pheatmap.r` and verified against R reference renders.

## Install

```bash
pip install pheatmap-python
```

Or, for a local development checkout:

```bash
git clone https://github.com/Bio-Babel/pheatmap-python.git
cd pheatmap_py
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np, pandas as pd
from pheatmap import pheatmap

test = pd.DataFrame(
    np.random.default_rng(1).standard_normal((20, 10)),
    index=[f"Gene{i+1}" for i in range(20)],
    columns=[f"Test{i+1}" for i in range(10)],
)

pheatmap(test)
```

See `tutorials/pheatmap_tutorial.ipynb` for a full walk-through.

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve
```
