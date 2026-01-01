# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
from anndata import AnnData
from loguru import logger

from ..computing.homology import compute_boundary_matrix_data
from ..data.boundary import BoundaryMatrixD0, BoundaryMatrixD1
from ..data.metadata import ScloopMeta
from ..data.types import Diameter_t


def compute_boundary_matrix_d1(
    adata: AnnData,
    meta: ScloopMeta,
    thresh: Diameter_t | None = None,
    verbose: bool = False,
    **nei_kwargs,
) -> BoundaryMatrixD1:
    assert meta.preprocess
    assert meta.preprocess.num_vertices
    (
        result,
        edge_ids,
        trig_ids,
        edge_diameters,
        _,
        _,
    ) = compute_boundary_matrix_data(
        adata=adata, meta=meta, thresh=thresh, **nei_kwargs
    )
    edge_ids_flat = np.array(edge_ids, dtype=np.int64).flatten()
    edge_diams_flat = np.array(edge_diameters, dtype=float)
    edge_ids_1d, uniq_idx = np.unique(edge_ids_flat, return_index=True)
    row_simplex_diams = edge_diams_flat[uniq_idx]
    edge_ids_reindex = np.searchsorted(edge_ids_1d, edge_ids)
    num_triangles = len(trig_ids)
    values = np.tile([1, -1, 1], num_triangles).tolist()

    bm_d1 = BoundaryMatrixD1(
        num_vertices=meta.preprocess.num_vertices,
        data=(
            edge_ids_reindex.flatten().tolist(),
            np.repeat(np.expand_dims(np.arange(num_triangles), 1), 3, axis=1)
            .flatten()
            .tolist(),
            values,
        ),
        shape=(len(edge_ids_1d), num_triangles),
        row_simplex_ids=edge_ids_1d.tolist(),
        col_simplex_ids=trig_ids,
        row_simplex_diams=row_simplex_diams.tolist(),
        col_simplex_diams=result.triangle_diameters,
    )

    if verbose:
        logger.info(
            f"Boundary matrix (dim 1) built: edges x triangles = "
            f"{bm_d1.shape[0]} x {bm_d1.shape[1]}"
        )
    return bm_d1


def compute_boundary_matrix_d0(
    boundary_matrix_d1: BoundaryMatrixD1,
    num_vertices: int,
    vertex_ids: list[int],
    verbose: bool = False,
) -> BoundaryMatrixD0:
    vertex_lookup = {
        int(vertex_id): row_idx for row_idx, vertex_id in enumerate(vertex_ids)
    }
    edges = boundary_matrix_d1.row_simplex_decode

    one_rows, one_cols, one_values = [], [], []
    for col_idx, e in enumerate(edges):
        u, v = e[0], e[1]

        one_rows.append(vertex_lookup[u])
        one_cols.append(col_idx)
        one_values.append(-1)

        one_rows.append(vertex_lookup[v])
        one_cols.append(col_idx)
        one_values.append(1)

    bm_d0 = BoundaryMatrixD0(
        num_vertices=num_vertices,
        data=(one_rows, one_cols, one_values),
        shape=(len(vertex_ids), boundary_matrix_d1.shape[0]),
        row_simplex_ids=vertex_ids,
        col_simplex_ids=boundary_matrix_d1.row_simplex_ids,
        row_simplex_diams=np.zeros(len(vertex_ids)).tolist(),
        col_simplex_diams=boundary_matrix_d1.row_simplex_diams,
    )

    if verbose:
        logger.info(
            f"Boundary matrix (dim 0) built: vertices x edges = "
            f"{bm_d0.shape[0]} x {bm_d0.shape[1]}"
        )
    return bm_d0
