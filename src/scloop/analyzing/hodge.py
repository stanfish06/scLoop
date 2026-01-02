# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import time

import numpy as np
from loguru import logger
from rich.progress import Progress
from scipy.sparse import csr_matrix

from ..computing.hodge_decomposition import (
    compute_hodge_eigendecomposition,
    compute_hodge_matrix,
)
from ..computing.matching import loops_to_edge_mask
from ..data.analysis_containers import BootstrapAnalysis, HodgeAnalysis
from ..data.base_components import LoopClass
from ..data.boundary import BoundaryMatrixD0, BoundaryMatrixD1
from ..data.constants import (
    DEFAULT_MAXITER_EIGENDECOMPOSITION,
    DEFAULT_N_HODGE_COMPONENTS,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
)
from ..data.metadata import ScloopMeta
from ..data.types import Count_t, Index_t, Percent_t
from ..data.utils import loops_masks_to_edges_masks


def compute_hodge_analysis(
    idx_track: Index_t,
    track_id: Index_t,  # usually same as idx_track
    bootstrap_data: BootstrapAnalysis,
    selected_loop_classes: list[LoopClass | None],
    boundary_matrix_d0: BoundaryMatrixD0,
    boundary_matrix_d1: BoundaryMatrixD1,
    meta: ScloopMeta,
    values_vertices: np.ndarray,
    life_pct: Percent_t | None = None,
    n_hodge_components: int = DEFAULT_N_HODGE_COMPONENTS,
    normalized: bool = True,
    n_neighbors_edge_embedding: Count_t = DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    weight_hodge: Percent_t = 0.5,
    half_window: int = 2,
    verbose: bool = False,
    progress: Progress | None = None,
    timeout_eigendecomposition: float = DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    maxiter_eigendecomposition: int | None = DEFAULT_MAXITER_EIGENDECOMPOSITION,
) -> None:
    assert idx_track in bootstrap_data.loop_tracks
    track = bootstrap_data.loop_tracks[idx_track]

    if life_pct is None:
        if meta.bootstrap is not None and meta.bootstrap.life_pct is not None:
            life_pct = meta.bootstrap.life_pct
        else:
            raise ValueError("life_pct not provided and not found in metadata")

    assert life_pct is not None
    assert idx_track < len(selected_loop_classes)
    loop_class = selected_loop_classes[idx_track]
    assert loop_class is not None

    birth_t = loop_class.birth
    death_t = loop_class.death
    thresh_t = birth_t + (death_t - birth_t) * life_pct

    start_time = time.perf_counter()

    task_step = None
    if progress is not None:
        task_step = progress.add_task("Computing Hodge matrix...", total=None)

    if verbose:
        logger.info("Computing Hodge matrix")

    # Prepare input matrices for compute_hodge_matrix
    d1_rows, d1_cols, d1_vals = boundary_matrix_d0.data
    bd1 = csr_matrix((d1_vals, (d1_rows, d1_cols)), shape=boundary_matrix_d0.shape)
    d2_rows, d2_cols, d2_vals = boundary_matrix_d1.data
    bd2_full = csr_matrix((d2_vals, (d2_rows, d2_cols)), shape=boundary_matrix_d1.shape)
    triangle_diams = np.array(boundary_matrix_d1.col_simplex_diams)

    hodge_matrix_d1 = compute_hodge_matrix(
        boundary_matrix_d0=bd1,
        boundary_matrix_d1=bd2_full,
        triangle_diams=triangle_diams,
        thresh=thresh_t,
        normalized=normalized,
    )

    if progress is not None and task_step is not None:
        progress.update(task_step, description="Computing Hodge eigendecomposition...")

    if verbose:
        logger.info("Computing Hodge eigendecomposition")

    result = compute_hodge_eigendecomposition(
        hodge_matrix=hodge_matrix_d1,
        timeout=timeout_eigendecomposition,
        n_components=n_hodge_components,
        maxiter=maxiter_eigendecomposition,
    )

    if result is None:
        logger.warning(f"Eigendecomposition failed for track {idx_track}")
        if progress is not None and task_step is not None:
            progress.remove_task(task_step)
        return

    eigenvalues, eigenvectors = result

    track.hodge_analysis = HodgeAnalysis(
        hodge_eigenvalues=eigenvalues.tolist(),
        hodge_eigenvectors=eigenvectors.T.tolist(),
    )

    source_loop_class = selected_loop_classes[idx_track]
    assert source_loop_class is not None

    if progress is not None and task_step is not None:
        progress.update(task_step, description="Analyzing loop classes...")

    if verbose:
        logger.info("Analyzing loop classes for track")

    bootstrap_data._analyze_track_loop_classes(
        idx_track=idx_track,
        source_loop_class=source_loop_class,
        values_vertices=values_vertices,
    )

    if progress is not None and task_step is not None:
        progress.update(task_step, description="Embedding edges...")

    if verbose:
        logger.info("Embedding edges")

    for loop in track.hodge_analysis.selected_loop_classes:
        assert loop.representatives is not None
        loops_mask, valid_indices_per_rep, edge_signs = loops_to_edge_mask(
            loops=loop.representatives,
            boundary_matrix_d1=boundary_matrix_d1,
            return_valid_indices=True,
            use_order=True,
        )
        loop.valid_edge_indices_per_rep = valid_indices_per_rep
        loop.edge_signs_per_rep = edge_signs
        track.hodge_analysis.edges_masks_loop_classes.append(
            loops_masks_to_edges_masks(loops_mask)
        )

    track.hodge_analysis._embed_edges(
        weight_hodge=weight_hodge,
        half_window=half_window,
    )
    try:
        if progress is not None and task_step is not None:
            progress.update(task_step, description="Smoothing edge embedding...")
        track.hodge_analysis._smoothening_edge_embedding(
            n_neighbors=n_neighbors_edge_embedding
        )
    except Exception as e:
        logger.warning(f"Edge smoothing failed: {e}")

    try:
        if progress is not None and task_step is not None:
            progress.update(task_step, description="Identify trajectories...")
        track.hodge_analysis._trajectory_identification()
    except Exception as e:
        logger.warning(f"Trajectory identification failed: {e}")

    if progress is not None and task_step is not None:
        progress.remove_task(task_step)

    if verbose:
        logger.success(
            f"Hodge analysis finished in {time.perf_counter() - start_time:.2f}s"
        )
