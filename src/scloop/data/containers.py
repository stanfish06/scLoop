# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from pydantic.dataclasses import dataclass
from abc import abstractmethod
from pydantic import ConfigDict, BaseModel, Field, field_validator, ValidationInfo
from scipy.spatial import distance_matrix
from .metadata import ScloopMeta
from .analysis_containers import BootstrapAnalysis, HodgeAnalysis
from anndata import AnnData
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csr_matrix
from .ripser_lib import ripser, get_boundary_matrix
from .loop_reconstruction import reconstruct_n_loop_representatives
from .types import IndexListDistMatrix, Diameter_t, Size_t, Index_t
from .utils import decode_edges, decode_triangles, encode_triangles_and_edges
from numba import jit
import numpy as np


class BoundaryMatrix(BaseModel):
    num_vertices: Size_t
    data: tuple[
        list[Index_t], list[Index_t]
    ]  # in coo format (row indices, col indices) of ones
    shape: tuple[Size_t, Size_t]
    row_simplex_ids: list[Index_t]
    col_simplex_ids: list[Index_t]
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
        assert self.meta.preprocess is not None
        assert self.meta.preprocess.embedding_method is not None
        boot_idx = None
        if bootstrap:
            boot_idx = np.random.choice(
                adata.shape[0], size=adata.shape[0], replace=True
            ).tolist()
        return radius_neighbors_graph(
            X=adata.obsm[f"X_{self.meta.preprocess.embedding_method}"],
            radius=thresh,
            **nei_kwargs,
        ), boot_idx

    def _compute_homology(
        self, adata: AnnData, thresh: Diameter_t | None = None, **nei_kwargs
    ) -> None:
        sparse_pairwise_distance_matrix, _ = self._compute_sparse_pairwise_distance(
            adata=adata, bootstrap=False, thresh=thresh, **nei_kwargs
        )
        result = ripser(
            distance_matrix=sparse_pairwise_distance_matrix,
            modulus=2,
            dim_max=1,
            threshold=thresh,
            do_cocyles=True,
        )
        self.persistence_diagram = result.births_and_deaths_by_dim

    def _compute_boundary_matrix(
        self, adata: AnnData, thresh: Diameter_t | None = None, **nei_kwargs
    ) -> None:
        assert self.meta.preprocess
        assert self.meta.preprocess.num_vertices
        sparse_pairwise_distance_matrix, _ = self._compute_sparse_pairwise_distance(
            adata=adata, bootstrap=False, thresh=thresh, **nei_kwargs
        )
        result = get_boundary_matrix(sparse_pairwise_distance_matrix, thresh)
        edge_ids, trig_ids = encode_triangles_and_edges(
            np.array(result.triangle_vertices), self.meta.preprocess.num_vertices
        )
        self.boundary_matrix_d1 = BoundaryMatrixD1(
            num_vertices=self.meta.preprocess.num_vertices,
            data=([], []),
            shape=(0, 0),
            row_simplex_ids=[],
            col_simplex_ids=[],
            col_simplex_diams=result.traingle_diameters,
        )

    def _compute_loop_representatives(self):
        pass
