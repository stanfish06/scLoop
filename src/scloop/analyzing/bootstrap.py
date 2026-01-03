# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np
from anndata import AnnData
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rich.progress import Progress

from ..computing.homology import compute_persistence_diagram_and_cocycles
from ..computing.loops import compute_loop_representatives
from ..computing.matching import (
    check_homological_equivalence,
    compute_geometric_distance,
)
from ..data.analysis_containers import LoopMatch
from ..data.base_components import LoopClass
from ..data.boundary import BoundaryMatrixD1
from ..data.constants import (
    DEFAULT_EXTRA_DIAM_EQUIVALENCE,
    DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
    DEFAULT_K_YEN,
    DEFAULT_LIFE_PCT,
    DEFAULT_LOOP_DIST_METHOD,
    DEFAULT_N_COCYCLES_USED,
    DEFAULT_N_FORCE_DEVIATE,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_PAIRS_CHECK_EQUIVALENCE,
    DEFAULT_N_REPS_PER_LOOP,
    DEFAULT_NOISE_SCALE,
)
from ..data.metadata import ScloopMeta
from ..data.types import LoopDistMethod
from ..data.utils import nearest_neighbor_per_row


class BootstrapResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    idx_bootstrap: int
    persistence_diagram: list | None
    cocycles: list | None
    indices_resample: list[int]
    loop_classes: list[LoopClass | None]
    matches: dict[int, list[LoopMatch]]


def run_single_bootstrap(
    idx_bootstrap: int,
    adata: AnnData,
    meta: ScloopMeta,
    original_loop_classes: list[LoopClass | None],
    original_boundary_matrix_d1: BoundaryMatrixD1,
    thresh: float | None = None,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    top_k: int = 1,
    n_reps_per_loop: int = DEFAULT_N_REPS_PER_LOOP,
    life_pct: float = DEFAULT_LIFE_PCT,
    n_cocycles_used: int = DEFAULT_N_COCYCLES_USED,
    n_force_deviate: int = DEFAULT_N_FORCE_DEVIATE,
    k_yen: int = DEFAULT_K_YEN,
    loop_lower_t_pct: float = 2.5,
    loop_upper_t_pct: float = 97.5,
    k_neighbors_check_equivalence: int = DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
    method_geometric_equivalence: LoopDistMethod = DEFAULT_LOOP_DIST_METHOD,
    n_pairs_check_equivalence: int = DEFAULT_N_PAIRS_CHECK_EQUIVALENCE,
    extra_diameter_homology_equivalence: float = DEFAULT_EXTRA_DIAM_EQUIVALENCE,
    filter_column_homology_equivalence: bool = True,
    **kwargs,
) -> BootstrapResult:
    (
        persistence_diagram,
        cocycles,
        indices_resample,
        sparse_pairwise_distance_matrix,
    ) = compute_persistence_diagram_and_cocycles(
        adata=adata,
        meta=meta,
        thresh=thresh,
        bootstrap=True,
        noise_scale=noise_scale,
        **kwargs,
    )

    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None
    assert indices_resample is not None

    embedding = np.array(adata.obsm[f"X_{meta.preprocess.embedding_method}"])

    bootstrap_loop_classes = compute_loop_representatives(
        embedding=embedding,
        pairwise_distance_matrix=sparse_pairwise_distance_matrix,
        persistence_diagram=persistence_diagram[1],
        cocycles=cocycles[1],
        boundary_matrix_d1=original_boundary_matrix_d1,
        vertex_ids=indices_resample,
        top_k=top_k,
        n_reps_per_loop=n_reps_per_loop,
        life_pct=life_pct,
        n_cocycles_used=n_cocycles_used,
        n_force_deviate=n_force_deviate,
        k_yen=k_yen,
        loop_lower_t_pct=loop_lower_t_pct,
        loop_upper_t_pct=loop_upper_t_pct,
        bootstrap=True,
    )

    n_original = len(original_loop_classes)
    n_bootstrap = len(bootstrap_loop_classes)

    matches: dict[int, list[LoopMatch]] = {}
    if n_original == 0 or n_bootstrap == 0:
        return BootstrapResult(
            idx_bootstrap=idx_bootstrap,
            persistence_diagram=persistence_diagram,
            cocycles=cocycles,
            indices_resample=indices_resample,
            loop_classes=bootstrap_loop_classes,
            matches=matches,
        )

    pairwise_geo_dist = np.full((n_original, n_bootstrap), np.nan)

    for i, src_loop in enumerate(original_loop_classes):
        if src_loop is None or src_loop.representatives is None:
            continue
        src_coords = src_loop.coordinates_vertices_representatives
        if src_coords is None:
            continue

        for j, tgt_loop in enumerate(bootstrap_loop_classes):
            if tgt_loop is None or tgt_loop.representatives is None:
                continue
            tgt_coords = tgt_loop.coordinates_vertices_representatives
            assert tgt_coords is not None

            dist = compute_geometric_distance(
                source_coords_list=src_coords,
                target_coords_list=tgt_coords,
                method=method_geometric_equivalence,
                n_workers=1,
            )
            pairwise_geo_dist[i, j] = dist

    neighbor_indices, neighbor_distances = nearest_neighbor_per_row(
        pairwise_geo_dist, k_neighbors_check_equivalence
    )

    for si in range(n_original):
        src_loop = original_loop_classes[si]
        if src_loop is None or src_loop.representatives is None:
            continue

        source_lifetime = src_loop.death - src_loop.birth
        source_loop_death = src_loop.death

        for k in range(k_neighbors_check_equivalence):
            tj = neighbor_indices[si, k]
            if tj < 0:
                continue

            tgt_loop = bootstrap_loop_classes[tj]
            if tgt_loop is None or tgt_loop.representatives is None:
                continue

            target_lifetime = tgt_loop.death - tgt_loop.birth
            target_loop_death = tgt_loop.death
            max_lifetime = max(source_lifetime, target_lifetime)

            max_column_diameter = None
            if filter_column_homology_equivalence:
                max_column_diameter = (
                    max(source_loop_death, target_loop_death)
                    + float(extra_diameter_homology_equivalence) * max_lifetime
                )

            is_equivalent = check_homological_equivalence(
                source_loops=src_loop.representatives,
                target_loops=tgt_loop.representatives,
                boundary_matrix_d1=original_boundary_matrix_d1,
                n_pairs_check=n_pairs_check_equivalence,
                max_column_diameter=max_column_diameter,
            )

            if is_equivalent:
                if si not in matches:
                    matches[si] = []
                matches[si].append(
                    LoopMatch(
                        idx_bootstrap=idx_bootstrap,
                        target_class_idx=int(tj),
                        geometric_distance=float(neighbor_distances[si, k]),
                        neighbor_rank=int(k),
                    )
                )

    return BootstrapResult(
        idx_bootstrap=idx_bootstrap,
        persistence_diagram=persistence_diagram,
        cocycles=cocycles,
        indices_resample=indices_resample,
        loop_classes=bootstrap_loop_classes,
        matches=matches,
    )


def run_bootstrap_pipeline(
    n_bootstrap: int,
    adata: AnnData,
    meta: ScloopMeta,
    original_loop_classes: list[LoopClass | None],
    original_boundary_matrix_d1: BoundaryMatrixD1,
    n_max_workers: int = DEFAULT_N_MAX_WORKERS,
    verbose: bool = False,
    progress: Progress | None = None,
    **kwargs,
) -> list[BootstrapResult]:
    results: list[BootstrapResult] = []

    ExecutorClass = ThreadPoolExecutor

    with ExecutorClass(max_workers=n_max_workers) as executor:
        tasks = {}
        for i in range(n_bootstrap):
            task = executor.submit(
                run_single_bootstrap,
                idx_bootstrap=i,
                adata=adata,
                meta=meta,
                original_loop_classes=original_loop_classes,
                original_boundary_matrix_d1=original_boundary_matrix_d1,
                **kwargs,
            )
            tasks[task] = i

        task_id = None
        if progress:
            task_id = progress.add_task("Working...", total=n_bootstrap)

        for task in as_completed(tasks):
            i = tasks[task]
            try:
                res = task.result()
                results.append(res)
                if verbose:
                    logger.success(f"Bootstrap {i + 1}/{n_bootstrap} finished")
            except Exception as e:
                logger.error(f"Bootstrap {i + 1}/{n_bootstrap} failed: {e}")

            if progress and task_id is not None:
                progress.advance(task_id)

    results.sort(key=lambda x: x.idx_bootstrap)
    return results
