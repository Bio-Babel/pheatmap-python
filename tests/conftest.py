"""Shared pytest fixtures for pheatmap_py tests.

Fixtures are backed by R-generated ``.npz``/``.json`` files under
``tests/fixtures/``.  Regenerate them with
``Rscript tests/fixtures/build_fixtures.R`` against the same R 4.5.x /
pheatmap 1.0.13 environment used for the executable baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_npz(name: str) -> dict[str, np.ndarray]:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"fixture missing: {path.name} — run build_fixtures.R")
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def _load_json(name: str) -> Any:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"fixture missing: {path.name} — run build_fixtures.R")
    with path.open("r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def test_matrix() -> np.ndarray:
    """A deterministic 20x10 matrix shared across tests."""
    data = _load_npz("test_matrix.npz")
    return np.asarray(data["matrix"], dtype=float)


@pytest.fixture(scope="session")
def hclust_complete_euclidean() -> dict[str, np.ndarray]:
    """R ``hclust(dist(m), 'complete')`` applied to ``test_matrix``."""
    return _load_npz("hclust_complete_euclidean.npz")


@pytest.fixture(scope="session")
def hclust_correlation_rows() -> dict[str, np.ndarray]:
    """R row clustering with the pheatmap-specific correlation distance."""
    return _load_npz("hclust_correlation_rows.npz")


@pytest.fixture(scope="session")
def hclust_minkowski_rows() -> dict[str, np.ndarray]:
    """R ``hclust(dist(m, 'minkowski', p=3), 'average')``."""
    return _load_npz("hclust_minkowski_rows.npz")


@pytest.fixture(scope="session")
def annotation_palette_seed3453() -> dict[str, Any]:
    """Palette emitted by R's ``dscale(...)`` under ``set.seed(3453)``."""
    return _load_json("annotation_palette_seed3453.json")
