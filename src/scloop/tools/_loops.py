# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

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


def find_loops(
    adata: AnnData,
    *,
    thresh: float | None = None,
    top_k: int | None = None,
) -> None:
    meta = _get_scloop_meta(adata)
    hd: HomologyData = HomologyData(meta=meta)  # type: ignore[call-arg]

    sparse_dist_mat = hd._compute_homology(adata=adata, thresh=thresh)  # type: ignore[attr-defined]
    hd._compute_boundary_matrix(adata=adata, thresh=thresh)  # type: ignore[attr-defined]

    top_k_val = top_k if top_k is not None else 0
    hd._compute_loop_representatives(  # type: ignore[attr-defined]
        pairwise_distance_matrix=sparse_dist_mat,
        top_k=top_k_val,
    )
    adata.uns["scloop"] = hd
