# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Sequence

import numpy as np
from anndata import AnnData

from ..data.containers import HomologyData
from ..data.metadata import ScloopMeta


def _get_scloop_meta(adata: AnnData) -> ScloopMeta:
    if "scloop_meta" not in adata.uns:
        raise ValueError("scloop_meta not found in adata.uns. Run prepare_adata first.")
    meta = adata.uns["scloop_meta"]
    if isinstance(meta, dict):
        meta = ScloopMeta(**meta)
    return meta


def _select_loop_indices(persistence_diagram: Sequence[np.ndarray], top_k: int | None):
    births, deaths = persistence_diagram
    pers = np.array(deaths) - np.array(births)
    if top_k is None or top_k >= len(pers):
        return list(range(len(pers)))
    return list(np.argsort(pers)[::-1][:top_k])


def find_loops(
    adata: AnnData,
    *,
    thresh: float | None = None,
    top_k: int | None = None,
    n_reps_per_loop: int = 8,
    life_pct: float = 0.1,
    n_force_deviate: int = 4,
    loop_lower_pct: float = 5.0,
    loop_upper_pct: float = 95.0,
    n_max_cocycles: int = 10,
) -> HomologyData:
    """Compute homology, boundary matrix, and loop representatives; store in adata.uns['scloop']."""
    meta = _get_scloop_meta(adata)
    hd: HomologyData = HomologyData(meta=meta)  # type: ignore[call-arg]

    hd._compute_homology(adata=adata, thresh=thresh)  # type: ignore[attr-defined]
    hd._compute_boundary_matrix(adata=adata, thresh=thresh)  # type: ignore[attr-defined]

    loop_indices = _select_loop_indices(hd.persistence_diagram[1], top_k)  # type: ignore[attr-defined]
    for idx in loop_indices:
        hd._compute_loop_representatives(  # type: ignore[attr-defined]
            loop_idx=idx,
            n=n_reps_per_loop,
            life_pct=life_pct,
            n_force_deviate=n_force_deviate,
            n_reps_per_loop=n_reps_per_loop,
            loop_lower_pct=loop_lower_pct,
            loop_upper_pct=loop_upper_pct,
            n_max_cocycles=n_max_cocycles,
        )

    adata.uns["scloop"] = hd
    return hd
