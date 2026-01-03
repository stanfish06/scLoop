# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from anndata import AnnData
from loguru import logger
from pydantic import Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..data.constants import (
    DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
    DEFAULT_MAX_COLUMNS_BOUNDARY_MATRIX,
    DEFAULT_MAXITER_EIGENDECOMPOSITION,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_HODGE_COMPONENTS,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    SCLOOP_META_UNS_KEY,
    SCLOOP_UNS_KEY,
)
from ..data.containers import HomologyData
from ..data.metadata import ScloopMeta
from ..data.types import Index_t, NonZeroCount_t, Percent_t, PositiveFloat, Size_t
from ..preprocessing.downsample import sample
from ..utils.logging import LogDisplay

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
    tightness_loops: Percent_t = 0,
    n_candidates: NonZeroCount_t = 1,
    n_bootstrap: Size_t = DEFAULT_N_BOOTSTRAP,
    n_check_per_candidate: NonZeroCount_t = 1,
    n_max_workers: NonZeroCount_t = DEFAULT_N_MAX_WORKERS,
    verbose: bool = False,
    max_log_messages: int | None = None,
    use_parallel: bool = False,
    kwargs_bootstrap: dict | None = None,
    kwargs_loop_test: dict | None = None,
    *,
    threshold_boundary: PositiveFloat | None = None,
    max_columns_boundary_matrix: NonZeroCount_t = DEFAULT_MAX_COLUMNS_BOUNDARY_MATRIX,
    auto_shrink_boundary_matrix: bool = True,
    auto_shrink_factor: Percent_t = 0.9,
    **kwargs,
) -> None:
    use_log_display = verbose and max_log_messages is not None
    log_display_ctx = None
    progress_main = None
    console = Console()

    if use_log_display:
        progress_main = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        log_display_ctx = LogDisplay(
            maxlen=max_log_messages, progress=progress_main, console=console
        )
        log_display_ctx.__enter__()

    try:
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
        hd._compute_boundary_matrix_d1(
            adata=adata, thresh=boundary_thresh, verbose=verbose
        )
        logger.info(f"Boundary matrix computed with threshold {boundary_thresh}")
        assert hd.boundary_matrix_d1 is not None
        assert meta.preprocess is not None
        if hd.boundary_matrix_d1.shape[1] > max_columns_boundary_matrix:
            logger.warning(
                f"Boundary matrix has more than {max_columns_boundary_matrix} columns. Downstream computation could be slow"
            )
            if auto_shrink_boundary_matrix:
                logger.info(
                    f"Autoshrink boundary matrix by re-downsample the data: dropping {(1 - auto_shrink_factor) * 100:.1f}% cells each attempt"
                )
                assert meta.preprocess.kwargs_downsample is not None
                kwargs_downsample = meta.preprocess.kwargs_downsample.copy()
                n_current = kwargs_downsample.pop("n")
                embedding_downsample = kwargs_downsample.get("embedding_method")
                assert n_current is not None
                if meta.preprocess.indices_downsample is not None:
                    n_current = int(n_current * auto_shrink_factor)
                n_downsample_final = n_current
                while (
                    hd.boundary_matrix_d1.shape[1] > max_columns_boundary_matrix
                    and n_current > 0
                ):
                    logger.info(
                        f"Downsampling to {n_current} cells using {embedding_downsample} embedding"
                    )
                    indices_downsample = sample(
                        adata=adata, n=n_current, **kwargs_downsample
                    )
                    meta.preprocess.indices_downsample = indices_downsample
                    hd._compute_boundary_matrix_d1(
                        adata=adata, thresh=boundary_thresh, verbose=verbose
                    )
                    n_downsample_final = n_current
                    n_current = int(n_current * auto_shrink_factor)
                kwargs_downsample["n"] = n_downsample_final
                meta.preprocess.kwargs_downsample = kwargs_downsample
                # re-compute homology with the new downsample indices
                sparse_dist_mat = hd._compute_homology(
                    adata=adata, thresh=threshold_homology
                )

        assert meta.preprocess.embedding_method is not None

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
            k_neighbors_check_equivalence=DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
            n_max_workers=n_max_workers,
            life_pct=tightness_loops,
            verbose=verbose,
            progress_main=progress_main,
            use_log_display=use_log_display,
            use_parallel=use_parallel,
            **(kwargs_bootstrap or {}),
        )

        """
        ========= statistcal tests =========
        - fisher exact test loop presence
        - gamma test of persistence
        ====================================
        """
        assert hd.bootstrap_data is not None
        hd._test_loops(**(kwargs_loop_test or {}))
        adata.uns[SCLOOP_UNS_KEY] = hd

    finally:
        if use_log_display and log_display_ctx:
            log_display_ctx.__exit__(None, None, None)


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
    max_log_messages: int | None = None,
    timeout_eigendecomposition: float = DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    maxiter_eigendecomposition: int | None = DEFAULT_MAXITER_EIGENDECOMPOSITION,
    compute_gene_trends: bool = True,
    gene_trend_genes: list[str] | None = None,
    gene_trend_confidence_level: float = 0.95,
    **kwargs_loop_analysis: Any,
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
    max_log_messages : int | None
        If set, use live log display showing last N messages with progress bar.
    **kwargs_edge_embedding
        Additional keyword arguments for edge embedding:
        - weight_hodge : float (default 0.5)
            Weight for Hodge embedding vs gradient (0-1). Higher = more Hodge.
        - half_window : int (default 2)
            Half window size for along-loop smoothing. 0 disables smoothing.
    """
    use_log_display = verbose and max_log_messages is not None
    log_display_ctx = None
    progress = None
    console = Console()

    if use_log_display:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        log_display_ctx = LogDisplay(
            maxlen=max_log_messages, progress=progress, console=console
        )
        log_display_ctx.__enter__()

    try:
        if SCLOOP_UNS_KEY not in adata.uns:
            raise ValueError("Run find_loops() first")

        if key_values not in adata.obs.columns:
            raise ValueError(f"{key_values} not found in adata.obs")

        values_vertices = np.array(adata.obs[key_values])

        coordinates_vertices = None
        if compute_gene_trends:
            assert adata.uns[SCLOOP_META_UNS_KEY].preprocess is not None
            coordinates_vertices = np.array(
                adata.obsm[
                    f"X_{adata.uns[SCLOOP_META_UNS_KEY].preprocess.embedding_method}"
                ]
            )

        gene_expression_matrix = None
        gene_names = None
        if compute_gene_trends:
            if gene_trend_genes is not None:
                adata_genes = adata[:, gene_trend_genes]
                gene_names = gene_trend_genes
            else:
                adata_genes = adata
                gene_names = adata.var_names.tolist()

            X = adata_genes.X
            gene_expression_matrix = X.toarray() if hasattr(X, "toarray") else X

        hd: HomologyData = adata.uns[SCLOOP_UNS_KEY]

        if hd.boundary_matrix_d0 is None:
            hd._compute_boundary_matrix_d0(verbose=verbose)

        assert hd.bootstrap_data is not None
        track_ids_avail = list(hd.bootstrap_data.loop_tracks.keys())
        if track_ids is None:
            track_ids = track_ids_avail

        if not use_log_display:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            )

        with progress:
            task_main = progress.add_task("Analyzing loops...", total=len(track_ids))

            for track_id in track_ids:
                if track_id not in track_ids_avail:
                    progress.advance(task_main)
                    continue

                progress.update(task_main, description=f"Analyzing track {track_id}...")

                hd._compute_hodge_analysis_for_track(
                    idx_track=track_id,
                    values_vertices=values_vertices,
                    coordinates_vertices=coordinates_vertices,
                    n_hodge_components=n_hodge_components,
                    normalized=normalized,
                    n_neighbors_edge_embedding=n_neighbors_edge_embedding,
                    verbose=verbose,
                    progress=progress,
                    timeout_eigendecomposition=timeout_eigendecomposition,
                    maxiter_eigendecomposition=maxiter_eigendecomposition,
                    compute_gene_trends=compute_gene_trends,
                    gene_expression_matrix=gene_expression_matrix,
                    gene_names=gene_names,
                    gene_trend_confidence_level=gene_trend_confidence_level,
                    **kwargs_loop_analysis,
                )
                progress.advance(task_main)

    finally:
        if use_log_display and log_display_ctx:
            log_display_ctx.__exit__(None, None, None)
