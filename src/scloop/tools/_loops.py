# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Annotated

from anndata import AnnData
from pydantic import Field

from ..data.analysis_containers import BootstrapAnalysis
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
    threshold_boundary: Annotated[float, Field(ge=0)] | None = None,
    tightness_loops: Annotated[float, Field(ge=0, le=1)] = 0,
    n_candidates: Annotated[int, Field(ge=1)] = 1,
    n_bootstrap: Annotated[int, Field(ge=0)] = 10,
    n_check_per_candidate: Annotated[int, Field(ge=1)] = 1,
    n_max_workers: Annotated[int, Field(ge=1)] = 8,
    verbose: bool = False,
) -> None:
    meta = _get_scloop_meta(adata)
    hd: HomologyData = HomologyData(meta=meta)
    sparse_dist_mat = hd._compute_homology(adata=adata, thresh=threshold_homology)
    boundary_thresh = threshold_boundary
    if boundary_thresh is None:
        boundary_thresh = threshold_homology
    hd._compute_boundary_matrix(adata=adata, thresh=boundary_thresh, verbose=verbose)
    hd._compute_loop_representatives(
        pairwise_distance_matrix=sparse_dist_mat,
        top_k=n_candidates,
        life_pct=tightness_loops,
    )
    """
    ========= bootstrap =========
    - resample data
    - find loops in resamples
    - map loops to original loops
    =============================
    """
    hd._bootstrap(
        adata=adata,
        n_bootstrap=n_bootstrap,
        thresh=threshold_homology,
        top_k=n_candidates * n_check_per_candidate,
        k_neighbors_check_equivalence=n_check_per_candidate,
        n_max_workers=n_max_workers,
        life_pct=tightness_loops,
        verbose=verbose,
    )

    """
    ========= statistcal tests =========
    - fisher exact test loop presence
    - gamma test of persistence
    ====================================
    """
    assert hd.bootstrap_data is not None
    bootstrap_data: BootstrapAnalysis = hd.bootstrap_data
    bootstrap_data.fisher_test_presence(method_pval_correction="benjamini-hochberg")
    adata.uns["scloop"] = hd
