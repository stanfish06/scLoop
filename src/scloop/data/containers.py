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
from .metadata import ScloopMeta
from .types import Diameter_t, Index_t, IndexListDistMatrix, Size_t
from .utils import decode_edges, decode_triangles


class BoundaryMatrix(BaseModel):
    num_vertices: Size_t
    data: tuple[
        list[Index_t], list[Index_t]
    ]  # in coo format (row indices, col indices) of ones
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
    loop_representatives: list[list[np.ndarray]] | None = None
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
        persistence_diagram, _, _ = compute_persistence_diagram_and_cocycles(
            adata=adata,
            meta=self.meta,
            thresh=thresh,
            bootstrap=bootstrap,
            **nei_kwargs,
        )
        self.persistence_diagram = persistence_diagram

    def _compute_boundary_matrix(
        self, adata: AnnData, thresh: Diameter_t | None = None, **nei_kwargs
    ) -> None:
        assert self.meta.preprocess
        assert self.meta.preprocess.num_vertices
        result, *_ = compute_boundary_matrix_data(
            adata=adata, meta=self.meta, thresh=thresh, **nei_kwargs
        )
        self.boundary_matrix_d1 = BoundaryMatrixD1(
            num_vertices=self.meta.preprocess.num_vertices,
            data=([], []),
            shape=(0, 0),
            row_simplex_ids=[],
            col_simplex_ids=[],
            row_simplex_diams=[],
            col_simplex_diams=result.triangle_diameters,
        )

    def _compute_loop_representatives(self):
        pass
