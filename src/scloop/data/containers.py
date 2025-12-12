# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from abc import abstractmethod

import numpy as np
from anndata import AnnData
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass
from scipy.sparse import csr_matrix

from ..computing.homology import (
    compute_boundary_matrix_data,
    compute_persistence_diagram_and_cocycles,
    compute_sparse_pairwise_distance,
)
from .analysis_containers import BootstrapAnalysis, HodgeAnalysis
from .loop_reconstruction import reconstruct_n_loop_representatives
from .metadata import ScloopMeta
from .types import Diameter_t, Index_t, IndexListDistMatrix, Size_t
from .utils import decode_edges, decode_triangles


class BoundaryMatrix(BaseModel):
    num_vertices: Size_t
    data: tuple[list, list]  # in coo format (row indices, col indices) of ones
    shape: tuple[Size_t, Size_t]
    row_simplex_ids: list[Index_t]
    col_simplex_ids: list[Index_t]
    row_simplex_diams: list[Diameter_t]
    col_simplex_diams: list[Diameter_t]

    @field_validator(
        "row_simplex_ids", "col_simplex_ids", "col_simplex_diams", mode="before"
    )
    @classmethod
    def validate_fields(cls, v: list[Index_t], info: ValidationInfo):
        shape = info.data.get("shape")
        assert shape
        if info.field_name == "row_simplex_ids":
            if len(v) != shape[0]:
                raise ValueError(
                    "Length of row ids does not match the number of rows of the matrix"
                )
        elif info.field_name in ["col_simplex_ids", "col_simplex_diams"]:
            if len(v) != shape[1]:
                raise ValueError(
                    f"Length of {info.field_name} does not match the number of columns of the matrix"
                )
        return v

    @abstractmethod
    def row_simplex_decode(self) -> list:
        """
        From simplex id (row) to vertex ids
        """
        pass

    @abstractmethod
    def col_simplex_decode(self) -> list:
        """
        From simplex id (column) to vertex ids
        """
        pass


class BoundaryMatrixD1(BoundaryMatrix):
    data: tuple[list[list[Index_t]], list[list[Index_t]]]

    def row_simplex_decode(self) -> list[tuple[Index_t, Index_t]]:
        return decode_edges(np.array(self.row_simplex_ids), self.num_vertices)

    def col_simplex_decode(self) -> list[tuple[Index_t, Index_t, Index_t]]:
        return decode_triangles(np.array(self.col_simplex_ids), self.num_vertices)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HomologyData:
    """
    store core homology data and associated analysis data
    """

    meta: ScloopMeta
    persistence_diagram: list[np.ndarray] | None = None
    loop_representatives: list[list[list[int]]] | None = None
    cocycles: list | None = None
    pairwise_distance_matrix: csr_matrix | None = None
    selected_vertex_indices: list[int] | None = None
    boundary_matrix_d1: BoundaryMatrixD1 | None = None
    bootstrap_data: BootstrapAnalysis | None = None
    hodge_data: HodgeAnalysis | None = None

    def _compute_sparse_pairwise_distance(
        self,
        adata: AnnData,
        bootstrap: bool = False,
        thresh: Diameter_t | None = None,
        **nei_kwargs,
    ) -> tuple[csr_matrix, IndexListDistMatrix | None]:
        return compute_sparse_pairwise_distance(
            adata=adata,
            meta=self.meta,
            bootstrap=bootstrap,
            thresh=thresh,
            **nei_kwargs,
        )

    def _compute_homology(
        self,
        adata: AnnData,
        thresh: Diameter_t | None = None,
        bootstrap: bool = False,
        **nei_kwargs,
    ) -> None:
        (
            persistence_diagram,
            cocycles,
            vertex_indices,
            sparse_pairwise_distance_matrix,
        ) = compute_persistence_diagram_and_cocycles(
            adata=adata,
            meta=self.meta,
            thresh=thresh,
            bootstrap=bootstrap,
            **nei_kwargs,
        )
        self.persistence_diagram = persistence_diagram
        self.cocycles = cocycles
        self.pairwise_distance_matrix = sparse_pairwise_distance_matrix
        self.selected_vertex_indices = vertex_indices

    def _compute_boundary_matrix(
        self, adata: AnnData, thresh: Diameter_t | None = None, **nei_kwargs
    ) -> None:
        assert self.meta.preprocess
        assert self.meta.preprocess.num_vertices
        (
            result,
            edge_ids,
            trig_ids,
            sparse_pairwise_distance_matrix,
            vertex_indices,
        ) = compute_boundary_matrix_data(
            adata=adata, meta=self.meta, thresh=thresh, **nei_kwargs
        )
        self.pairwise_distance_matrix = sparse_pairwise_distance_matrix
        self.selected_vertex_indices = vertex_indices
        edge_ids_1d = np.array(edge_ids).flatten()
        # reindex edges (also keep as colllection of triplets, easier to subset later)
        edge_ids_reindex = np.searchsorted(edge_ids_1d, edge_ids)
        edge_diameters = decode_edges(edge_ids_1d, self.meta.preprocess.num_vertices)
        edge_diameters = [
            sparse_pairwise_distance_matrix[i, j] for i, j in edge_diameters
        ]
        self.boundary_matrix_d1 = BoundaryMatrixD1(
            num_vertices=self.meta.preprocess.num_vertices,
            data=(
                edge_ids_reindex.tolist(),
                np.repeat(
                    np.expand_dims(np.arange(edge_ids_reindex.shape[0]), 1), 3, axis=1
                ).tolist(),
            ),
            shape=(len(edge_ids_1d), len(trig_ids)),
            row_simplex_ids=edge_ids_1d.tolist(),
            col_simplex_ids=trig_ids,
            row_simplex_diams=edge_diameters,
            col_simplex_diams=result.triangle_diameters,
        )

    def _compute_loop_representatives(
        self,
        loop_idx: int,
        n: int = 8,
        life_pct: float = 0.1,
        n_force_deviate: int = 4,
        n_reps_per_loop: int = 8,
        loop_lower_pct: float = 5,
        loop_upper_pct: float = 95,
        n_max_cocycles: int = 10,
    ):
        assert self.persistence_diagram is not None
        assert self.cocycles is not None
        assert self.pairwise_distance_matrix is not None

        births, deaths = self.persistence_diagram[1]
        loop_birth = float(births[loop_idx])
        loop_death = float(deaths[loop_idx])

        dm = self.pairwise_distance_matrix.tocoo()
        edge_weights: dict[tuple[int, int], float] = {}
        for i, j, w in zip(dm.row.tolist(), dm.col.tolist(), dm.data.tolist()):
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key not in edge_weights or w < edge_weights[key]:
                edge_weights[key] = float(w)
        if not edge_weights:
            return [], []

        edges = list(edge_weights.keys())
        edge_births = np.array([edge_weights[e] for e in edges], dtype=float)

        loops, dists = reconstruct_n_loop_representatives(
            cocycles_dim1=self.cocycles[1][loop_idx],
            edges=edges,
            edge_births=edge_births,
            loop_birth=loop_birth,
            loop_death=loop_death,
            n=n,
            life_pct=life_pct,
            n_force_deviate=n_force_deviate,
            n_reps_per_loop=n_reps_per_loop,
            loop_lower_pct=loop_lower_pct,
            loop_upper_pct=loop_upper_pct,
            n_max_cocycles=n_max_cocycles,
        )

        if self.loop_representatives is None:
            self.loop_representatives = []
        while len(self.loop_representatives) <= loop_idx:
            self.loop_representatives.append([])
        self.loop_representatives[loop_idx] = loops
        return loops, dists
