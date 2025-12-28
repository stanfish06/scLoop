# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from anndata import AnnData
from pydantic import Field

from ..data.constants import (
    DEFAULT_N_HODGE_COMPONENTS,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    SCLOOP_META_UNS_KEY,
    SCLOOP_UNS_KEY,
)
from ..data.containers import HomologyData
from ..data.metadata import ScloopMeta
from ..data.types import Index_t, Percent_t, PositiveFloat, Size_t

__all__ = ["find_loops", "analyze_loops"]


def _get_scloop_meta(adata: AnnData) -> ScloopMeta:
    if SCLOOP_META_UNS_KEY not in adata.uns:
        raise ValueError("scloop_meta not found in adata.uns. Run prepare_adata first.")
    meta = adata.uns[SCLOOP_META_UNS_KEY]
    if isinstance(meta, dict):
        meta = ScloopMeta(**meta)
    return meta


def find_loops(
    adata: AnnData,
    threshold_homology: PositiveFloat | None = None,
    threshold_boundary: PositiveFloat | None = None,
    tightness_loops: Percent_t = 0,
    n_candidates: Annotated[int, Field(ge=1)] = 1,
    n_bootstrap: Size_t = 10,
    n_check_per_candidate: Annotated[int, Field(ge=1)] = 1,
    n_max_workers: Annotated[int, Field(ge=1)] = DEFAULT_N_MAX_WORKERS,
    verbose: bool = False,
) -> None:
    meta = _get_scloop_meta(adata)
    if meta.bootstrap is None:
        from ..data.metadata import BootstrapMeta

        meta.bootstrap = BootstrapMeta()
    meta.bootstrap.life_pct = tightness_loops
    hd: HomologyData = HomologyData(meta=meta)
    sparse_dist_mat = hd._compute_homology(adata=adata, thresh=threshold_homology)
    boundary_thresh = threshold_boundary
    if boundary_thresh is None:
        boundary_thresh = threshold_homology
    hd._compute_boundary_matrix_d1(adata=adata, thresh=boundary_thresh, verbose=verbose)
    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None
    import numpy as np

    embedding = np.array(adata.obsm[f"X_{meta.preprocess.embedding_method}"])
    hd._compute_loop_representatives(
        embedding=embedding,
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
    hd._test_loops(method_pval_correction="benjamini-hochberg")
    adata.uns[SCLOOP_UNS_KEY] = hd


def analyze_loops(
    adata: AnnData,
    track_ids: list[Index_t] | None = None,
    key_values: str = "dpt_pseudotime",
    n_hodge_components: Annotated[int, Field(ge=1)] = DEFAULT_N_HODGE_COMPONENTS,
    n_neighbors_edge_embedding: Annotated[
        int, Field(ge=1)
    ] = DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    normalized: bool = True,
    verbose: bool = False,
    **kwargs_edge_embedding: Any,
) -> None:
    """Analyze loops using Hodge decomposition and edge embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with scloop results from find_loops().
    track_ids : list[Index_t] | None
        List of track IDs to analyze. If None, analyze all tracks.
    key_values : str
        Key in adata.obs for vertex values (e.g., pseudotime).
    n_hodge_components : int
        Number of Hodge eigenvector components to compute.
    n_neighbors_edge_embedding : int
        Number of neighbors for KNN smoothing of edge embedding.
    normalized : bool
        Whether to use normalized Hodge Laplacian.
    verbose : bool
        Whether to print progress messages.
    **kwargs_edge_embedding
        Additional keyword arguments for edge embedding:
        - weight_hodge : float (default 0.5)
            Weight for Hodge embedding vs gradient (0-1). Higher = more Hodge.
        - half_window : int (default 2)
            Half window size for along-loop smoothing. 0 disables smoothing.
    """
    if SCLOOP_UNS_KEY not in adata.uns:
        raise ValueError("Run find_loops() first")

    if key_values not in adata.obs.columns:
        raise ValueError(f"{key_values} not found in adata.obs")

    values_vertices = np.array(adata.obs[key_values])

    hd: HomologyData = adata.uns[SCLOOP_UNS_KEY]

    if hd.boundary_matrix_d0 is None:
        hd._compute_boundary_matrix_d0(verbose=verbose)

    assert hd.bootstrap_data is not None
    track_ids_avail = list(hd.bootstrap_data.loop_tracks.keys())
    if track_ids is None:
        track_ids = track_ids_avail

    for track_id in track_ids:
        if track_id not in track_ids_avail:
            continue
        hd._compute_hodge_analysis_for_track(
            idx_track=track_id,
            values_vertices=values_vertices,
            n_hodge_components=n_hodge_components,
            normalized=normalized,
            n_neighbors_edge_embedding=n_neighbors_edge_embedding,
            verbose=verbose,
            **kwargs_edge_embedding,
        )
