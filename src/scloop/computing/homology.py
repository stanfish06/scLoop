# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import radius_neighbors_graph

from ..data.metadata import ScloopMeta
from ..data.ripser_lib import (  # type: ignore[import-not-found]
    get_boundary_matrix,
    ripser,
)
from ..data.types import Diameter_t, IndexListDistMatrix, LoopDistMethod
from ..data.utils import encode_triangles_and_edges
from ..utils.distance_metrics.frechet_py import compute_pairwise_loop_frechet
from ..utils.linear_algebra_gf2.m4ri_lib import (
    solve_multiple_gf2,  # type: ignore[import-not-found]
)

if TYPE_CHECKING:
    from ..data.containers import BoundaryMatrixD1


def compute_sparse_pairwise_distance(
    adata: AnnData,
    meta: ScloopMeta,
    bootstrap: bool = False,
    noise_scale: float = 1e-3,
    thresh: Diameter_t | None = None,
    **nei_kwargs,
) -> tuple[csr_matrix, IndexListDistMatrix | None]:
    # important, default is binary graph
    nei_kwargs.setdefault("mode", "distance")
    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None
    emb = adata.obsm[f"X_{meta.preprocess.embedding_method}"]
    selected_indices = (
        meta.preprocess.indices_downsample
        if meta.preprocess.indices_downsample is not None
        else list(range(emb.shape[0]))
    )
    X = emb[selected_indices]
    boot_idx = None
    if bootstrap:
        sample_idx = np.random.choice(
            len(selected_indices), size=len(selected_indices), replace=True
        ).tolist()
        boot_idx = [selected_indices[i] for i in sample_idx]
        std_X = np.std(X, axis=0)
        X = X[sample_idx] + np.random.normal(scale=std_X * noise_scale, size=X.shape)
    else:
        boot_idx = selected_indices
    return (
        radius_neighbors_graph(
            X=X,
            radius=thresh,
            **nei_kwargs,
        ),
        boot_idx,
    )


def compute_persistence_diagram_and_cocycles(
    adata: AnnData,
    meta: ScloopMeta,
    thresh: Diameter_t | None = None,
    bootstrap: bool = False,
    noise_scale: float = 1e3,
    **nei_kwargs,
) -> tuple[list, list, IndexListDistMatrix | None, csr_matrix]:
    def _cap_infinite_deaths(diagrams: list, cap: float | None) -> list:
        if cap is None or not np.isfinite(cap):
            return diagrams
        capped = []
        for dim_pd in diagrams:
            if len(dim_pd) < 2:
                capped.append(dim_pd)
                continue
            births = np.asarray(dim_pd[0])
            deaths = np.asarray(dim_pd[1])
            deaths = np.where(np.isinf(deaths), cap, deaths)
            capped.append([births, deaths])
        return capped

    sparse_pairwise_distance_matrix, boot_idx = compute_sparse_pairwise_distance(
        adata=adata,
        meta=meta,
        bootstrap=bootstrap,
        noise_scale=noise_scale,
        thresh=thresh,
        **nei_kwargs,
    )
    result = ripser(
        distance_matrix=sparse_pairwise_distance_matrix.tocoo(copy=False),
        modulus=2,
        dim_max=1,
        threshold=thresh,
        do_cocycles=True,
    )
    return (
        _cap_infinite_deaths(result.births_and_deaths_by_dim, thresh),
        result.cocycles_by_dim,
        boot_idx,
        sparse_pairwise_distance_matrix,
    )


def compute_boundary_matrix_data(
    adata: AnnData, meta: ScloopMeta, thresh: Diameter_t | None = None, **nei_kwargs
) -> tuple:
    """
    Compute both D0 and D1 boundary matrices
    """
    assert meta.preprocess is not None
    assert meta.preprocess.num_vertices is not None
    sparse_pairwise_distance_matrix, vertex_indices = compute_sparse_pairwise_distance(
        adata=adata, meta=meta, bootstrap=False, thresh=thresh, **nei_kwargs
    )
    result = get_boundary_matrix(sparse_pairwise_distance_matrix.tocoo(), thresh)
    triangles_local = np.asarray(result.triangle_vertices, dtype=np.int64)
    if len(triangles_local) == 0:
        edge_ids, trig_ids, edge_diameters, vertex_indices_np = [], [], [], np.array([])
    else:
        if vertex_indices is None:
            assert sparse_pairwise_distance_matrix.shape is not None
            vertex_indices_np = np.arange(sparse_pairwise_distance_matrix.shape[0])
        else:
            vertex_indices_np = np.asarray(vertex_indices, dtype=np.int64)
        # important: must convert triangle vertex ids to global indices
        triangles = vertex_indices_np[triangles_local]
        # CRITICAL: sort triangle vertices to ensure consistent orientation
        # Without this, d1 @ d2 != 0 when downsampling is used
        triangles = np.sort(triangles, axis=1)
        # NOTE: edges and triangles are encoded based on the total number of vertices, not the downsampled number
        edge_ids, trig_ids = encode_triangles_and_edges(
            triangles, meta.preprocess.num_vertices
        )
        edge_diameters = []
        for tri_local in triangles_local:
            i0, i1, i2 = int(tri_local[0]), int(tri_local[1]), int(tri_local[2])
            edge_diameters.extend(
                [
                    sparse_pairwise_distance_matrix[i0, i1],
                    sparse_pairwise_distance_matrix[i0, i2],
                    sparse_pairwise_distance_matrix[i1, i2],
                ]
            )
    return (
        result,
        edge_ids,
        trig_ids,
        edge_diameters,
        sparse_pairwise_distance_matrix,
        vertex_indices_np.tolist(),
    )


def compute_loop_homological_equivalence(
    boundary_matrix_d1: "BoundaryMatrixD1",
    loop_mask_a: np.ndarray,
    loop_mask_b: np.ndarray,
    n_pairs_check: int = 3,
    max_column_diameter: float | None = None,
) -> tuple[list, list]:
    """
    Parameters
    ---------
    loop_mask_a: np.ndarray
        Boolean mask of shape (n_a, n_edges); True where edge (row) is in the loop
    loop_mask_b: np.ndarray
        Boolean mask of shape (n_b, n_edges)
    max_column_diameter: float | None
        If provided, restrict the boundary matrix to columns (triangles) with diameter
        no larger than this value.
    """
    assert loop_mask_a.shape[1] == boundary_matrix_d1.shape[0]
    assert loop_mask_b.shape[1] == boundary_matrix_d1.shape[0]

    # in F2, sum is just xor
    loop_sums = loop_mask_a[:, None, :] ^ loop_mask_b[None, :, :]
    loop_sums = loop_sums.reshape(-1, loop_sums.shape[-1])
    if loop_sums.shape[0] == 0:
        return [], []
    n_pairs_check = min(n_pairs_check, loop_sums.shape[0])

    one_ridx_A = np.asarray(boundary_matrix_d1.data[0])
    one_cidx_A = np.asarray(boundary_matrix_d1.data[1])
    nrow_A = boundary_matrix_d1.shape[0]
    ncol_A = boundary_matrix_d1.shape[1]

    col_diams = np.asarray(boundary_matrix_d1.col_simplex_diams, dtype=float)
    if max_column_diameter is not None:
        cols_keep = np.flatnonzero(col_diams <= max_column_diameter)
        if cols_keep.size == 0:
            return [], []
        mask = np.isin(one_cidx_A, cols_keep)
        one_ridx_A = one_ridx_A[mask]
        one_cidx_A = one_cidx_A[mask]
        # reindex columns
        col_reindex = -np.ones(ncol_A, dtype=int)
        col_reindex[cols_keep] = np.arange(cols_keep.size, dtype=int)
        one_cidx_A = col_reindex[one_cidx_A]
        ncol_A = cols_keep.size
        col_diams = col_diams[cols_keep]

    # still have redundant columns, drop large triangles
    # this can be caused by 2D holes but they are too expensive to check
    if ncol_A > nrow_A:
        cols_keep_sorted = np.argsort(col_diams)[:nrow_A]

        mask = np.isin(one_cidx_A, cols_keep_sorted)
        one_ridx_A = one_ridx_A[mask]
        one_cidx_A = one_cidx_A[mask]

        col_reindex = -np.ones(ncol_A, dtype=int)
        col_reindex[cols_keep_sorted] = np.arange(nrow_A, dtype=int)
        one_cidx_A = col_reindex[one_cidx_A]
        ncol_A = nrow_A

    one_idx_b_list = [
        np.flatnonzero(loop_sums[i]).astype(int).tolist() for i in range(n_pairs_check)
    ]
    results, solutions = solve_multiple_gf2(
        one_ridx_A=one_ridx_A.tolist(),
        one_cidx_A=one_cidx_A.tolist(),
        nrow_A=nrow_A,
        ncol_A=ncol_A,
        one_idx_b_list=one_idx_b_list,
    )

    return results, solutions


def compute_loop_geometric_distance(
    source_coords_list: list[list[list[float]]] | list[np.ndarray],
    target_coords_list: list[list[list[float]]] | list[np.ndarray],
    method: LoopDistMethod = "hausdorff",
) -> np.ndarray:
    if len(source_coords_list) == 0 or len(target_coords_list) == 0:
        return np.array([np.nan])

    match method:
        case "frechet":
            try:
                distances_arr = compute_pairwise_loop_frechet(
                    source_coords_list, target_coords_list
                )
                return distances_arr
            except Exception:
                return np.full(
                    len(source_coords_list) * len(target_coords_list), np.nan
                )
        case "hausdorff":
            distances = []
            for source_coords in source_coords_list:
                for target_coords in target_coords_list:
                    try:
                        dist = max(
                            directed_hausdorff(source_coords, target_coords)[0],
                            directed_hausdorff(target_coords, source_coords)[0],
                        )
                        distances.append(dist)
                    except (ValueError, IndexError):
                        distances.append(np.nan)
            return np.array(distances)
        case _:
            return np.full(len(source_coords_list) * len(target_coords_list), np.nan)
