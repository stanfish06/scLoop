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
    boot_idx = None
    X = adata.obsm[f"X_{meta.preprocess.embedding_method}"]
    if bootstrap:
        boot_idx = np.random.choice(
            adata.shape[0], size=adata.shape[0], replace=True
        ).tolist()
        X = X[boot_idx] + np.random.normal(scale=noise_scale, size=X.shape)
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
) -> tuple[list[np.ndarray], IndexListDistMatrix | None, csr_matrix]:
    sparse_pairwise_distance_matrix, boot_idx = compute_sparse_pairwise_distance(
        adata=adata, meta=meta, bootstrap=bootstrap, thresh=thresh, **nei_kwargs
    )
    result = ripser(
        distance_matrix=sparse_pairwise_distance_matrix,
        modulus=2,
        dim_max=1,
        threshold=thresh,
        do_cocyles=True,
    )
    return result.births_and_deaths_by_dim, boot_idx, sparse_pairwise_distance_matrix


def compute_boundary_matrix_data(
    adata: AnnData, meta: ScloopMeta, thresh: Diameter_t | None = None, **nei_kwargs
):
    assert meta.preprocess is not None
    assert meta.preprocess.num_vertices is not None
    sparse_pairwise_distance_matrix, _ = compute_sparse_pairwise_distance(
        adata=adata, meta=meta, bootstrap=False, thresh=thresh, **nei_kwargs
    )
    result = get_boundary_matrix(sparse_pairwise_distance_matrix, thresh)
    edge_ids, trig_ids = encode_triangles_and_edges(
        np.array(result.triangle_vertices), meta.preprocess.num_vertices
    )
    return result, edge_ids, trig_ids, sparse_pairwise_distance_matrix
