# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph

from ..data.metadata import ScloopMeta
from ..data.ripser_lib import get_boundary_matrix, ripser  # type: ignore[import-not-found]
from ..data.types import Diameter_t, IndexListDistMatrix
from ..data.utils import encode_triangles_and_edges
from ..utils.linear_algebra_gf2.m4ri_lib import solve_multiple_gf2  # type: ignore[import-not-found]

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
        X = X[sample_idx] + np.random.normal(scale=noise_scale, size=X.shape)
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
        result.births_and_deaths_by_dim,
        result.cocycles_by_dim,
        boot_idx,
        sparse_pairwise_distance_matrix,
    )


def compute_boundary_matrix_data(
    adata: AnnData, meta: ScloopMeta, thresh: Diameter_t | None = None, **nei_kwargs
) -> tuple:
    assert meta.preprocess is not None
    assert meta.preprocess.num_vertices is not None
    sparse_pairwise_distance_matrix, vertex_indices = compute_sparse_pairwise_distance(
        adata=adata, meta=meta, bootstrap=False, thresh=thresh, **nei_kwargs
    )
    result = get_boundary_matrix(sparse_pairwise_distance_matrix.tocoo(), thresh)
    triangles = np.asarray(result.triangle_vertices, dtype=np.int64)
    if len(triangles) == 0:
        edge_ids, trig_ids = [], []
    else:
        edge_ids, trig_ids = encode_triangles_and_edges(
            triangles, meta.preprocess.num_vertices
        )
    return result, edge_ids, trig_ids, sparse_pairwise_distance_matrix, vertex_indices


def compute_loop_homological_equivalence(
    boundary_matrix_d1: "BoundaryMatrixD1",
    loop_mask_a: np.ndarray,
    loop_mask_b: np.ndarray,
    n_pairs_check: int = 3,
) -> tuple[list, list]:
    """
    Parameters
    ---------
    loop_mask_a: np.ndarray
        Boolean mask of shape (n_a, n_edges); True where edge (row) is in the loop
    loop_mask_b: np.ndarray
        Boolean mask of shape (n_b, n_edges)
    """
    assert loop_mask_a.shape[1] == boundary_matrix_d1.shape[0]
    assert loop_mask_b.shape[1] == boundary_matrix_d1.shape[0]

    # in F2, sum is just xor
    loop_sums = loop_mask_a[:, None, :] ^ loop_mask_b[None, :, :]
    loop_sums = loop_sums.reshape(-1, loop_sums.shape[-1])
    if loop_sums.shape[0] == 0:
        return [], []
    n_pairs_check = min(n_pairs_check, loop_sums.shape[0])

    one_idx_b_list = [
        np.flatnonzero(loop_sums[i]).astype(int).tolist() for i in range(n_pairs_check)
    ]
    results, solutions = solve_multiple_gf2(
        one_ridx_A=boundary_matrix_d1.data[0],
        one_cidx_A=boundary_matrix_d1.data[1],
        nrow_A=boundary_matrix_d1.shape[0],
        ncol_A=boundary_matrix_d1.shape[1],
        one_idx_b_list=one_idx_b_list,
    )

    return results, solutions
