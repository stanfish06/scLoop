# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from abc import abstractmethod

import numpy as np
from anndata import AnnData
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass
from scipy.sparse import csr_matrix, triu

from ..computing.homology import (
    compute_boundary_matrix_data,
    compute_persistence_diagram_and_cocycles,
)
from .analysis_containers import BootstrapAnalysis, HodgeAnalysis
from .loop_reconstruction import reconstruct_n_loop_representatives
from .metadata import ScloopMeta
from .types import Diameter_t, Index_t, Size_t
from .utils import decode_edges, decode_triangles, extract_edges_from_coo


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
    persistence_diagram: list | None = None
    loop_representatives: list[list[list[int]]] = Field(default_factory=list)
    cocycles: list | None = None
    boundary_matrix_d1: BoundaryMatrixD1 | None = None
    bootstrap_data: BootstrapAnalysis | None = None
    hodge_data: HodgeAnalysis | None = None

    def _compute_homology(
        self,
        adata: AnnData,
        thresh: Diameter_t | None = None,
        bootstrap: bool = False,
        **nei_kwargs,
    ) -> csr_matrix:
        (
            persistence_diagram,
            cocycles,
            indices_resample,
            sparse_pairwise_distance_matrix,
        ) = compute_persistence_diagram_and_cocycles(
            adata=adata,
            meta=self.meta,
            thresh=thresh,
            bootstrap=bootstrap,
            **nei_kwargs,
        )
        if not bootstrap:
            self.persistence_diagram = persistence_diagram
            self.cocycles = cocycles
        else:
            assert self.bootstrap_data is not None
            self.bootstrap_data.persistence_diagrams.append(persistence_diagram)  # type: ignore[attr-defined]
            self.bootstrap_data.cocycles.append(cocycles)  # type: ignore[attr-defined]
            self.meta.bootstrap.indices_resample.append(indices_resample)  # type: ignore[attr-defined]
        return sparse_pairwise_distance_matrix

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
            _,
        ) = compute_boundary_matrix_data(
            adata=adata, meta=self.meta, thresh=thresh, **nei_kwargs
        )
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
        pairwise_distance_matrix: csr_matrix,
        vertex_ids: list[int],  # important, must be the indicies in original data
        top_k: int = 1,  # top k homology classes to compute representatives
        bootstrap: bool = False,
        idx_bootstrap: int = 0,
        n_reps_per_loop: int = 4,
        life_pct: float = 0.1,
        n_cocycles_used: int = 10,
        n_force_deviate: int = 4,
        k_yen: int = 8,
        loop_lower_t_pct: float = 5,
        loop_upper_t_pct: float = 95,
    ):
        if not bootstrap:
            assert self.persistence_diagram is not None
            assert self.cocycles is not None
            loop_births = np.array(self.persistence_diagram[1][0], dtype=np.float32)
            loop_deaths = np.array(self.persistence_diagram[1][1], dtype=np.float32)
            cocycles = self.cocycles[1]
        else:
            assert self.bootstrap_data is not None
            assert len(self.bootstrap_data.persistence_diagrams) > idx_bootstrap  # type: ignore[attr-defined]
            assert len(self.bootstrap_data.cocyles) > idx_bootstrap  # type: ignore[attr-defined]
            loop_births = np.array(
                self.bootstrap_data.persistence_diagrams[idx_bootstrap][1][0],
                dtype=np.float32,
            )  # type: ignore[attr-defined]
            loop_deaths = np.array(
                self.bootstrap_data.persistence_diagrams[idx_bootstrap][1][1],
                dtype=np.float32,
            )  # type: ignore[attr-defined]
            cocycles = self.bootstrap_data.cocyles[idx_bootstrap].cocycles[1]  # type: ignore[attr-defined]

        # get top k homology classes
        indices_top_k = np.argpartition(loop_deaths - loop_births, -top_k)[-top_k:]

        dm_upper = triu(pairwise_distance_matrix, k=1).tocoo()
        edges_array, edge_diameters = extract_edges_from_coo(
            dm_upper.row, dm_upper.col, dm_upper.data
        )

        if len(edges_array) == 0:
            return [], []

        # Initialize storage
        if not bootstrap:
            if self.loop_representatives is None:
                self.loop_representatives = []
            while len(self.loop_representatives) < len(indices_top_k):
                self.loop_representatives.append([])
        else:
            assert self.bootstrap_data is not None
            bootstrap_data = self.bootstrap_data
            while len(bootstrap_data.loop_representatives) <= idx_bootstrap:  # type: ignore[attr-defined]
                bootstrap_data.loop_representatives.append([])  # type: ignore[attr-defined]
            if len(bootstrap_data.loop_representatives[idx_bootstrap]) < len(
                indices_top_k
            ):  # type: ignore[attr-defined]
                bootstrap_data.loop_representatives[idx_bootstrap] = [
                    [] for _ in range(len(indices_top_k))
                ]  # type: ignore[attr-defined]

        for loop_idx, i in enumerate(indices_top_k):
            loop_birth: float = loop_births[i].item()
            loop_death: float = loop_deaths[i].item()
            loops_local, _ = reconstruct_n_loop_representatives(
                cocycles_dim1=cocycles[i],
                edges=edges_array,
                edge_diameters=edge_diameters,
                loop_birth=loop_birth,
                loop_death=loop_death,
                n=n_reps_per_loop,
                life_pct=life_pct,
                n_force_deviate=n_force_deviate,
                k_yen=k_yen,
                loop_lower_pct=loop_lower_t_pct,
                loop_upper_pct=loop_upper_t_pct,
                n_cocycles_used=n_cocycles_used,
            )

            loops = [[vertex_ids[v] for v in loop] for loop in loops_local]

            if not bootstrap:
                self.loop_representatives[loop_idx] = loops
            else:
                bootstrap_data.loop_representatives[idx_bootstrap][loop_idx] = loops  # type: ignore[attr-defined]

    def _bootstrap(
        self,
        adata: AnnData,
        n: int,
        thresh: Diameter_t | None = None,
        noise_scale: float = 1e-3,
        n_reps_per_loop: int = 8,
        life_pct: float = 0.1,
        n_force_deviate: int = 4,
        loop_lower_pct: float = 5,
        loop_upper_pct: float = 95,
        n_max_cocycles: int = 10,
        verbose: bool = True,
        **nei_kwargs,
    ):
        pass
        # from tqdm import tqdm
        # from sklearn.neighbors import radius_neighbors_graph

        # assert self.meta.preprocess is not None
        # assert self.boundary_matrix_d1 is not None, "Run find_loops first"

        # self.bootstrap_data = BootstrapAnalysis(num_bootstraps=n)

        # emb_key = f"X_{self.meta.preprocess.embedding_method}"
        # X_orig = adata.obsm[emb_key]

        # if self.meta.preprocess.indices_downsample is not None:
        #     X_orig = X_orig[self.meta.preprocess.indices_downsample]

        # n_cells = X_orig.shape[0]

        # for boot_iter in tqdm(range(n), desc="Bootstrap", disable=not verbose):
        #     boot_indices = np.random.choice(n_cells, size=n_cells, replace=True)
        #     X_boot = X_orig[boot_indices] + np.random.normal(
        #         scale=noise_scale, size=X_orig.shape
        #     )

        #     sparse_dist_mat = radius_neighbors_graph(
        #         X=X_boot,
        #         radius=thresh,
        #         mode='distance',
        #         metric='euclidean',
        #         **nei_kwargs,
        #     )

        #     from .ripser_lib import ripser
        #     result = ripser(
        #         distance_matrix=sparse_dist_mat.tocoo(copy=False),
        #         modulus=2,
        #         dim_max=1,
        #         threshold=thresh,
        #         do_cocycles=True,
        #     )

        #     bootstrap_data = self.bootstrap_data
        #     bootstrap_data.persistence_diagrams.append(result.births_and_deaths_by_dim)
        #     bootstrap_data.cocycles.append(result.cocycles_by_dim)

        #     births_1, deaths_1 = result.births_and_deaths_by_dim[1]
        #     if len(births_1) == 0:
        #         bootstrap_data.loop_representatives.append([])
        #         continue

        #     n_loops_boot = len(births_1)
        #     boot_loops_all = []

        #     from scipy.sparse import triu
        #     dm_upper = triu(sparse_dist_mat, k=1).tocoo()

        #     for loop_idx in range(n_loops_boot):
        #         loop_birth = float(births_1[loop_idx])
        #         loop_death = float(deaths_1[loop_idx])

        #         edges_array, edge_births = extract_edges_from_coo(dm_upper.row, dm_upper.col, dm_upper.data)

        #         if len(edges_array) == 0:
        #             boot_loops_all.append([])
        #             continue

        #         edges = [(int(e[0]), int(e[1])) for e in edges_array]

        #         loops_local, _ = reconstruct_n_loop_representatives(
        #             cocycles_dim1=result.cocycles_by_dim[1][loop_idx],
        #             edges=edges,
        #             edge_births=edge_births,
        #             loop_birth=loop_birth,
        #             loop_death=loop_death,
        #             n=n_reps_per_loop,
        #             life_pct=life_pct,
        #             n_force_deviate=n_force_deviate,
        #             n_reps_per_loop=n_reps_per_loop,
        #             loop_lower_pct=loop_lower_pct,
        #             loop_upper_pct=loop_upper_pct,
        #             n_max_cocycles=n_max_cocycles,
        #         )

        #         loops = [
        #             [boot_indices[v] for v in loop]
        #             for loop in loops_local
        #         ]
        #         boot_loops_all.append(loops)

        #     bootstrap_data.loop_representatives.append(boot_loops_all)
