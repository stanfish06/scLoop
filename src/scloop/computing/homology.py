# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph

from ..data.metadata import ScloopMeta
from ..data.ripser_lib import get_boundary_matrix, ripser
from ..data.types import Diameter_t, IndexListDistMatrix
from ..data.utils import encode_triangles_and_edges


def compute_sparse_pairwise_distance(
    adata: AnnData,
    meta: ScloopMeta,
    bootstrap: bool = False,
    noise_scale: float = 1e-3,
    thresh: Diameter_t | None = None,
    **nei_kwargs,
) -> tuple[csr_matrix, IndexListDistMatrix | None]:
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
    **nei_kwargs,
) -> tuple[list[np.ndarray], list, IndexListDistMatrix | None, csr_matrix]:
    sparse_pairwise_distance_matrix, boot_idx = compute_sparse_pairwise_distance(
        adata=adata, meta=meta, bootstrap=bootstrap, thresh=thresh, **nei_kwargs
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
