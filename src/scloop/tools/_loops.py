# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Annotated

from anndata import AnnData
from pydantic import Field

from ..data.containers import HomologyData
from ..data.metadata import ScloopMeta

__all__ = ["find_loops"]


def _get_scloop_meta(adata: AnnData) -> ScloopMeta:
    if "scloop_meta" not in adata.uns:
        raise ValueError("scloop_meta not found in adata.uns. Run prepare_adata first.")
    meta = adata.uns["scloop_meta"]
    if isinstance(meta, dict):
        meta = ScloopMeta(**meta)
    return meta


def find_loops(
    adata: AnnData,
    threshold_homology: Annotated[float, Field(ge=0)] | None = None,
    n_candidates: Annotated[int, Field(ge=1)] = 1,
    n_bootstrap: Annotated[int, Field(ge=0)] = 10,
    n_check_per_candidate: Annotated[int, Field(ge=1)] = 1,
    verbose: bool = False,
) -> None:
    meta = _get_scloop_meta(adata)
    hd: HomologyData = HomologyData(meta=meta)

    sparse_dist_mat = hd._compute_homology(adata=adata, thresh=threshold_homology)
    hd._compute_boundary_matrix(adata=adata, thresh=threshold_homology)
    hd._compute_loop_representatives(
        pairwise_distance_matrix=sparse_dist_mat,
        top_k=n_candidates,
    )

    # hd._bootstrap(
    #     adata=adata,
    #     n_bootstrap=n_bootstrap,
    #     thresh=threshold_homology,
    #     top_k=n_candidates * n_check_per_candidate,
    #     verbose=verbose,
    # )
    adata.uns["scloop"] = hd
