# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from anndata import AnnData
from loguru import logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.sparse import csr_matrix, triu
from scipy.sparse.linalg import eigsh

from ..computing.homology import (
    compute_boundary_matrix_data,
    compute_loop_geometric_distance,
    compute_loop_homological_equivalence,
    compute_persistence_diagram_and_cocycles,
)
from .analysis_containers import (
    BootstrapAnalysis,
    LoopMatch,
    LoopTrack,
)
from .base_components import LoopClass
from .constants import (
    DEFAULT_N_HODGE_COMPONENTS,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
)
from .loop_reconstruction import reconstruct_n_loop_representatives
from .metadata import BootstrapMeta, ScloopMeta
from .types import (
    Count_t,
    Diameter_t,
    Index_t,
    IndexListDownSample,
    LoopDistMethod,
    MultipleTestCorrectionMethod,
    Percent_t,
    Size_t,
)
from .utils import (
    decode_edges,
    decode_triangles,
    extract_edges_from_coo,
    loop_vertices_to_edge_ids_with_signs,
    loops_masks_to_edges_masks,
    loops_to_coords,
    nearest_neighbor_per_row,
)


class BoundaryMatrix(BaseModel, ABC):
    num_vertices: Size_t
    data: tuple[
        list[Index_t], list[Index_t], list[int]
    ]  # in coo format (row indices, col indices, values)
    shape: tuple[Size_t, Size_t]
    row_simplex_ids: list[Index_t]
    col_simplex_ids: list[Index_t]
    row_simplex_diams: list[Diameter_t]
    col_simplex_diams: list[Diameter_t]

    @field_validator(
        "row_simplex_ids",
        "row_simplex_diams",
        "col_simplex_ids",
        "col_simplex_diams",
        mode="before",
    )
    @classmethod
    def validate_fields(cls, v: list[Index_t], info: ValidationInfo):
        shape = info.data.get("shape")
        assert shape
        if info.field_name in ["row_simplex_ids", "row_simplex_diams"]:
            if len(v) != shape[0]:
                raise ValueError(
                    f"Length of {info.field_name} does not match the number of rows of the matrix"
                )
        elif info.field_name in ["col_simplex_ids", "col_simplex_diams"]:
            if len(v) != shape[1]:
                raise ValueError(
                    f"Length of {info.field_name} does not match the number of columns of the matrix"
                )
        return v

    @property
    @abstractmethod
    def row_simplex_decode(self) -> list:
        """
        From simplex id (row) to vertex ids
        """
        pass

    @property
    @abstractmethod
    def col_simplex_decode(self) -> list:
        """
        From simplex id (column) to vertex ids
        """
        pass


class BoundaryMatrixD1(BoundaryMatrix):
    @property
    def row_simplex_decode(self) -> list[tuple[Index_t, Index_t]]:
        return decode_edges(np.array(self.row_simplex_ids), self.num_vertices)

    @property
    def col_simplex_decode(self) -> list[tuple[Index_t, Index_t, Index_t]]:
        return decode_triangles(np.array(self.col_simplex_ids), self.num_vertices)


class BoundaryMatrixD0(BoundaryMatrix):
    @property
    def row_simplex_decode(self) -> list[Index_t]:
        return self.row_simplex_ids

    @property
    def col_simplex_decode(self) -> list[tuple[Index_t, Index_t]]:
        return decode_edges(np.array(self.col_simplex_ids), self.num_vertices)


@dataclass
class HomologyData:
    """Store core homology data and associated analysis data

    Attributes
    ----------
    attribute_name : type
    Description of attribute.
    """

    meta: ScloopMeta
    persistence_diagram: list | None = None
    cocycles: list | None = None
    selected_loop_classes: list[LoopClass | None] = Field(default_factory=list)
    boundary_matrix_d1: BoundaryMatrixD1 | None = None
    boundary_matrix_d0: BoundaryMatrixD0 | None = None
    bootstrap_data: BootstrapAnalysis | None = None

    def _loops_to_edge_mask(
        self,
        loops: list[list[int]],
        return_valid_indices: bool = False,
        use_order: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[list[int]], list[np.ndarray]]:
        assert self.boundary_matrix_d1 is not None
        num_vertices = self.boundary_matrix_d1.num_vertices
        n_edges = self.boundary_matrix_d1.shape[0]

        edge_lookup = {
            int(edge_id): row_idx
            for row_idx, edge_id in enumerate(self.boundary_matrix_d1.row_simplex_ids)
        }

        dtype = np.int32 if use_order else bool
        mask = np.zeros((len(loops), n_edges), dtype=dtype)
        valid_indices_per_rep = []
        edge_signs_per_rep = []
        for idx, loop in enumerate(loops):
            edge_ids, edge_signs = loop_vertices_to_edge_ids_with_signs(
                np.asarray(loop, dtype=np.int64), num_vertices
            )
            valid_indices = []
            valid_signs = []
            seen_row_ids = set()
            order = 1  # 1-based traversal order
            for edge_idx, (eid, sign) in enumerate(zip(edge_ids, edge_signs)):
                row_id = edge_lookup.get(int(eid), -1)
                if row_id >= 0 and row_id not in seen_row_ids:
                    mask[idx, row_id] = order if use_order else True
                    order += 1
                    valid_indices.append(edge_idx)
                    valid_signs.append(sign)
                    seen_row_ids.add(row_id)
            valid_indices_per_rep.append(valid_indices)
            edge_signs_per_rep.append(np.array(valid_signs, dtype=np.int8))

        if return_valid_indices:
            return mask, valid_indices_per_rep, edge_signs_per_rep
        return mask

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
            assert self.meta.bootstrap is not None
            assert indices_resample is not None
            assert self.meta.bootstrap.indices_resample is not None

            self.bootstrap_data.persistence_diagrams.append(persistence_diagram)
            self.bootstrap_data.cocycles.append(cocycles)
            self.meta.bootstrap.indices_resample.append(indices_resample)
        return sparse_pairwise_distance_matrix

    def _compute_boundary_matrix_d1(
        self,
        adata: AnnData,
        thresh: Diameter_t | None = None,
        verbose: bool = False,
        **nei_kwargs,
    ) -> None:
        assert self.meta.preprocess
        assert self.meta.preprocess.num_vertices
        (
            result,
            edge_ids,
            trig_ids,
            edge_diameters,
            _,
            _,
        ) = compute_boundary_matrix_data(
            adata=adata, meta=self.meta, thresh=thresh, **nei_kwargs
        )
        edge_ids_flat = np.array(edge_ids, dtype=np.int64).flatten()
        edge_diams_flat = np.array(edge_diameters, dtype=float)
        edge_ids_1d, uniq_idx = np.unique(edge_ids_flat, return_index=True)
        row_simplex_diams = edge_diams_flat[uniq_idx]
        edge_ids_reindex = np.searchsorted(edge_ids_1d, edge_ids)
        num_triangles = len(trig_ids)
        values = np.tile([1, -1, 1], num_triangles).tolist()
        self.boundary_matrix_d1 = BoundaryMatrixD1(
            num_vertices=self.meta.preprocess.num_vertices,
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
                f"{self.boundary_matrix_d1.shape[0]} x {self.boundary_matrix_d1.shape[1]}"
            )

    @property
    def _original_vertex_ids(self):
        assert self.meta.preprocess is not None
        assert self.meta.preprocess.num_vertices is not None
        if self.meta.preprocess.indices_downsample is not None:
            vertex_ids: IndexListDownSample = self.meta.preprocess.indices_downsample
        else:
            vertex_ids = (
                np.arange(self.meta.preprocess.num_vertices).astype(np.int64).tolist()
            )
        # make sure only unique indices
        assert len(vertex_ids) == len(set(vertex_ids)), "vertex ids must be unqiue"
        return vertex_ids

    def _compute_boundary_matrix_d0(self, verbose: bool = False):
        assert self.boundary_matrix_d1 is not None
        assert self.meta.preprocess
        assert self.meta.preprocess.num_vertices

        # important, if downsampled, vertex indecies are no longer sorted
        vertex_ids = sorted(self._original_vertex_ids)
        vertex_lookup = {
            int(vertex_id): row_idx for row_idx, vertex_id in enumerate(vertex_ids)
        }
        edges = self.boundary_matrix_d1.row_simplex_decode

        one_rows, one_cols, one_values = [], [], []
        for col_idx, e in enumerate(edges):
            # e is a sorted tuple (u, v) from edge decoding
            # boundary is v - u, so u gets -1 and v gets +1
            u, v = e[0], e[1]

            one_rows.append(vertex_lookup[u])
            one_cols.append(col_idx)
            one_values.append(-1)

            one_rows.append(vertex_lookup[v])
            one_cols.append(col_idx)
            one_values.append(1)

        self.boundary_matrix_d0 = BoundaryMatrixD0(
            num_vertices=self.meta.preprocess.num_vertices,
            data=(one_rows, one_cols, one_values),
            shape=(len(vertex_ids), self.boundary_matrix_d1.shape[0]),
            row_simplex_ids=vertex_ids,
            col_simplex_ids=self.boundary_matrix_d1.row_simplex_ids,
            row_simplex_diams=np.zeros(len(vertex_ids)).tolist(),
            col_simplex_diams=self.boundary_matrix_d1.row_simplex_diams,
        )
        if verbose:
            logger.info(
                f"Boundary matrix (dim 0) built: vertices x edges = "
                f"{self.boundary_matrix_d0.shape[0]} x {self.boundary_matrix_d0.shape[1]}"
            )

    def _compute_hodge_matrix(
        self, thresh: Diameter_t, normalized: bool = True
    ) -> csr_matrix | None:
        if self.boundary_matrix_d0 is None or self.boundary_matrix_d1 is None:
            raise ValueError("Boundary matrices must be computed first.")

        d1_rows, d1_cols, d1_vals = self.boundary_matrix_d0.data
        bd1 = csr_matrix(
            (d1_vals, (d1_rows, d1_cols)), shape=self.boundary_matrix_d0.shape
        )

        d2_rows, d2_cols, d2_vals = self.boundary_matrix_d1.data
        bd2_full = csr_matrix(
            (d2_vals, (d2_rows, d2_cols)), shape=self.boundary_matrix_d1.shape
        )

        bd1_bd2 = bd1.dot(bd2_full)
        assert type(bd1_bd2) is csr_matrix
        if bd1_bd2.count_nonzero() != 0:
            raise ValueError(
                f"d1 @ d2 has {bd1_bd2.count_nonzero()} nonzero entries. "
                f"Simplex orientation is incorrect."
            )

        triangle_diams = np.array(self.boundary_matrix_d1.col_simplex_diams)
        cols_use = np.where(triangle_diams <= thresh)[0]

        if cols_use.size == 0:
            logger.warning(
                f"No triangles below threshold {thresh}. hodge_matrix_d1 is d1T*d1 only."
            )
            bd2 = csr_matrix(bd2_full.shape)
        else:
            bd2 = bd2_full[:, cols_use]

        if normalized:
            D2 = np.maximum(abs(bd2).sum(1), 1)
            D1 = 2 * (abs(bd1) @ D2)
            D3 = 1 / 3
            L1 = (bd1.T.multiply(D2).multiply(1 / D1.T)) @ bd1 + (
                (bd2 * D3) @ bd2.T
            ).multiply(1 / D2.T)
            L1 = L1.multiply(1 / np.sqrt(D2)).multiply(np.sqrt(D2).T)
            hodge_matrix_d1: csr_matrix = csr_matrix(L1)
        else:
            hodge_matrix_d1 = csr_matrix(bd1.transpose() @ bd1 + bd2 @ bd2.transpose())

        return hodge_matrix_d1

    def _compute_hodge_eigendecomposition(
        self, hodge_matrix: csr_matrix, n_components: int = 10
    ) -> tuple[np.ndarray, np.ndarray] | None:
        assert type(hodge_matrix) is csr_matrix
        assert hodge_matrix.shape is not None
        if hodge_matrix.shape[0] < 2:
            logger.warning("hodge_matrix too small for eigendecomposition (shape < 2).")
            return None

        k = min(n_components, hodge_matrix.shape[0] - 2)
        if k <= 0:
            logger.warning(f"Not enough dimensions for eigendecomposition (k={k}).")
            return None

        try:
            eigenvalues, eigenvectors = eigsh(hodge_matrix, k=k, which="SA", tol=1e-6)  # type: ignore[arg-type]
            sort_idx = np.argsort(eigenvalues)
            return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
        except Exception as e:
            logger.error(f"Eigendecomposition failed: {e}")
            return None

    def _compute_hodge_analysis_for_track(
        self,
        idx_track: Index_t,
        values_vertices: np.ndarray,
        life_pct: Percent_t | None = None,
        n_hodge_components: int = DEFAULT_N_HODGE_COMPONENTS,
        normalized: bool = True,
        n_neighbors_edge_embedding: Count_t = DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
        weight_hodge: Percent_t = 0.5,
        half_window: int = 2,
        verbose: bool = False,
    ) -> None:
        """Analyze a specific loop track

        Parameters
        ----------
        idx_track : Index_t
            Index of the loop track to analyze.
        values_vertices : np.ndarray
            Values at vertices (e.g., pseudotime) for computing gradients.
        life_pct : Percent_t | None
            Percentage of lifetime to use for threshold.
        n_hodge_components : int
            Number of Hodge eigenvector components to compute.
        normalized : bool
            Whether to use normalized Hodge Laplacian.
        n_neighbors_edge_embedding : Count_t
            Number of neighbors for KNN smoothing of edge embedding.
        weight_hodge : Percent_t
            Weight for Hodge embedding vs gradient (0-1). Higher = more Hodge.
        half_window : int
            Half window size for along-loop smoothing. 0 disables smoothing.
        verbose : bool
            Whether to print progress messages.

        Returns
        -------
        None
        """

        assert self.bootstrap_data is not None
        assert idx_track in self.bootstrap_data.loop_tracks

        track = self.bootstrap_data.loop_tracks[idx_track]

        if life_pct is None:
            if (
                self.meta.bootstrap is not None
                and self.meta.bootstrap.life_pct is not None
            ):
                life_pct = self.meta.bootstrap.life_pct
            else:
                raise ValueError("life_pct not provided and not found in metadata")
        assert life_pct is not None
        assert idx_track < len(self.selected_loop_classes)
        loop_class = self.selected_loop_classes[idx_track]
        assert loop_class is not None
        birth_t = loop_class.birth
        death_t = loop_class.death
        thresh_t = birth_t + (death_t - birth_t) * life_pct

        start_time = time.perf_counter()
        if verbose:
            logger.info("Computing Hodge matrix")
        hodge_matrix_d1 = self._compute_hodge_matrix(
            thresh=thresh_t, normalized=normalized
        )
        if hodge_matrix_d1 is None:
            logger.warning(f"Could not compute Hodge matrix for track {idx_track}")
            return

        if verbose:
            logger.info("Computing Hodge eigendecomposition")
        result = self._compute_hodge_eigendecomposition(
            hodge_matrix=hodge_matrix_d1,
            n_components=n_hodge_components,
        )

        if result is None:
            logger.warning(f"Eigendecomposition failed for track {idx_track}")
            return

        eigenvalues, eigenvectors = result

        from .analysis_containers import HodgeAnalysis

        track.hodge_analysis = HodgeAnalysis(
            hodge_eigenvalues=eigenvalues.tolist(),
            hodge_eigenvectors=eigenvectors.T.tolist(),  # needs transpose (columns are eig-vecs)
        )

        source_loop_class = self.selected_loop_classes[idx_track]
        assert source_loop_class is not None
        if verbose:
            logger.info("Analyzing loop classes for track")
        self.bootstrap_data._analyze_track_loop_classes(
            idx_track=idx_track,
            source_loop_class=source_loop_class,
            values_vertices=values_vertices,
        )

        """
        ============= edge embedding =============
        - compute edge masks for loops
        - embed edges using edge masks and hodge
        - guassian smooth edge embedding
        - trajectory discovery
        ==========================================
        """
        if verbose:
            logger.info("Embedding edges")
        for loop in track.hodge_analysis.selected_loop_classes:
            assert loop.representatives is not None
            loops_mask, valid_indices_per_rep, edge_signs = self._loops_to_edge_mask(
                loops=loop.representatives,
                return_valid_indices=True,
                use_order=True,
            )
            loop.valid_edge_indices_per_rep = valid_indices_per_rep
            loop.edge_signs_per_rep = edge_signs
            track.hodge_analysis.edges_masks_loop_classes.append(
                loops_masks_to_edges_masks(loops_mask)
            )

        track.hodge_analysis._embed_edges(
            weight_hodge=weight_hodge,
            half_window=half_window,
        )
        track.hodge_analysis._smoothening_edge_embedding(
            n_neighbors=n_neighbors_edge_embedding
        )
        if verbose:
            logger.success(
                f"Hodge analysis finished in {time.perf_counter() - start_time:.2f}s"
            )

    def _compute_loop_representatives(
        self,
        embedding: np.ndarray,
        pairwise_distance_matrix: csr_matrix,
        top_k: int | None = None,  # top k homology classes to compute representatives
        bootstrap: bool = False,
        idx_bootstrap: int = 0,
        n_reps_per_loop: int = 4,
        life_pct: Percent_t = 0.1,
        n_cocycles_used: int = 3,
        n_force_deviate: int = 4,
        k_yen: int = 8,
        loop_lower_t_pct: float = 2.5,
        loop_upper_t_pct: float = 97.5,
    ):
        assert pairwise_distance_matrix.shape is not None
        assert self.meta.preprocess is not None
        if not bootstrap:
            assert self.persistence_diagram is not None
            assert self.cocycles is not None
            loop_births = np.array(self.persistence_diagram[1][0], dtype=np.float32)
            loop_deaths = np.array(self.persistence_diagram[1][1], dtype=np.float32)
            cocycles = self.cocycles[1]
            vertex_ids = self._original_vertex_ids
        else:
            assert self.bootstrap_data is not None
            assert len(self.bootstrap_data.persistence_diagrams) > idx_bootstrap
            assert len(self.bootstrap_data.cocycles) > idx_bootstrap
            assert self.meta.bootstrap is not None
            assert self.meta.bootstrap.indices_resample is not None
            assert len(self.meta.bootstrap.indices_resample) > idx_bootstrap
            loop_births = np.array(
                self.bootstrap_data.persistence_diagrams[idx_bootstrap][1][0],
                dtype=np.float32,
            )
            loop_deaths = np.array(
                self.bootstrap_data.persistence_diagrams[idx_bootstrap][1][1],
                dtype=np.float32,
            )
            cocycles = self.bootstrap_data.cocycles[idx_bootstrap][1]
            vertex_ids = self.meta.bootstrap.indices_resample[idx_bootstrap]

        if loop_births.size == 0:
            return [], []
        if top_k is None:
            top_k = loop_births.size
        if top_k <= 0:
            return [], []
        top_k = min(top_k, loop_births.size)

        persistence = loop_deaths - loop_births
        indices_top_k = np.argsort(persistence)[::-1][:top_k]

        dm_upper = triu(pairwise_distance_matrix, k=1).tocoo()
        edges_array, edge_diameters = extract_edges_from_coo(
            dm_upper.row, dm_upper.col, dm_upper.data
        )

        if len(edges_array) == 0:
            return [], []

        if not bootstrap:
            while len(self.selected_loop_classes) < len(indices_top_k):
                self.selected_loop_classes.append(None)
        else:
            assert self.bootstrap_data is not None
            while len(self.bootstrap_data.selected_loop_classes) <= idx_bootstrap:
                self.bootstrap_data.selected_loop_classes.append([])
            while len(self.bootstrap_data.selected_loop_classes[idx_bootstrap]) < len(
                indices_top_k
            ):
                self.bootstrap_data.selected_loop_classes[idx_bootstrap].append(None)

        for loop_idx, i in enumerate(indices_top_k):
            loop_birth = loop_births[i].item()
            loop_death = loop_deaths[i].item()
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
            loops_coords = loops_to_coords(embedding=embedding, loops_vertices=loops)

            if not bootstrap:
                self.selected_loop_classes[loop_idx] = LoopClass(
                    rank=loop_idx,
                    birth=loop_birth,
                    death=loop_death,
                    cocycles=cocycles[i],
                    representatives=loops,
                    coordinates_vertices_representatives=loops_coords,
                )
            else:
                assert self.bootstrap_data is not None
                self.bootstrap_data.selected_loop_classes[idx_bootstrap][loop_idx] = (
                    LoopClass(
                        rank=loop_idx,
                        birth=loop_birth,
                        death=loop_death,
                        cocycles=cocycles[i],
                        representatives=loops,
                        coordinates_vertices_representatives=loops_coords,
                    )
                )

    def _get_loop_embedding(
        self,
        selector: Index_t | tuple[Index_t, Index_t],
        embedding_alt: np.ndarray | None = None,
        include_bootstrap: bool = True,
    ) -> list[list[list[float]]]:
        """
        Use embedding stored in LoopClass by default
        If emebdding_alt provided, use that instead
        """
        loops = []
        match selector:
            case int():
                assert selector < len(self.selected_loop_classes)
                loop_class = self.selected_loop_classes[selector]
                if loop_class is not None:
                    if embedding_alt is None:
                        if loop_class.coordinates_vertices_representatives is not None:
                            loops.extend(
                                loop_class.coordinates_vertices_representatives
                            )
                    else:
                        if loop_class.representatives is not None:
                            loops.extend(
                                loops_to_coords(
                                    embedding=embedding_alt,
                                    loops_vertices=loop_class.representatives,
                                )
                            )
                if include_bootstrap:
                    assert self.bootstrap_data is not None
                    loops.extend(
                        self.bootstrap_data._get_track_embedding(
                            idx_track=selector, embedding_alt=embedding_alt
                        )
                    )
            case tuple():
                assert self.bootstrap_data is not None
                assert selector[0] < len(self.bootstrap_data.selected_loop_classes)
                assert selector[1] < len(
                    self.bootstrap_data.selected_loop_classes[selector[0]]
                )
                loops.extend(
                    self.bootstrap_data._get_loop_embedding(
                        idx_bootstrap=selector[0],
                        idx_loop=selector[1],
                        embedding_alt=embedding_alt,
                    )
                )
        return loops

    def _assess_bootstrap_geometric_equivalence(
        self,
        source_class_idx: Index_t,
        target_class_idx: Index_t,
        idx_bootstrap: int = 0,
        method: LoopDistMethod = "hausdorff",
    ) -> tuple[int, int, float]:
        assert self.bootstrap_data is not None
        assert self.meta.preprocess is not None
        assert self.meta.preprocess.embedding_method is not None

        if idx_bootstrap >= len(self.bootstrap_data.selected_loop_classes):
            return (source_class_idx, target_class_idx, np.nan)
        if source_class_idx >= len(self.selected_loop_classes):
            return (source_class_idx, target_class_idx, np.nan)

        if target_class_idx >= len(
            self.bootstrap_data.selected_loop_classes[idx_bootstrap]
        ):
            return (source_class_idx, target_class_idx, np.nan)

        source_coords_list = self._get_loop_embedding(
            selector=source_class_idx, include_bootstrap=False
        )
        target_coords_list = self._get_loop_embedding(
            selector=(idx_bootstrap, target_class_idx), include_bootstrap=False
        )

        distances_arr = compute_loop_geometric_distance(
            source_coords_list, target_coords_list, method
        )
        mean_distance = float(np.nanmean(distances_arr))
        return (source_class_idx, target_class_idx, mean_distance)

    def _assess_bootstrap_homology_equivalence(
        self,
        source_class_idx: int,
        target_class_idx: int | None = None,
        idx_bootstrap: int = 0,
        n_pairs_check: int = 10,
    ) -> tuple[int, int, bool]:
        assert self.bootstrap_data is not None
        self._ensure_loop_tracks()
        if target_class_idx is None:
            target_class_idx = source_class_idx
        if idx_bootstrap >= len(self.bootstrap_data.selected_loop_classes):
            return (source_class_idx, target_class_idx, False)
        if source_class_idx >= len(self.selected_loop_classes):
            return (source_class_idx, target_class_idx, False)
        if target_class_idx >= len(
            self.bootstrap_data.selected_loop_classes[idx_bootstrap]
        ):
            return (source_class_idx, target_class_idx, False)

        source_loop_class = self.selected_loop_classes[source_class_idx]
        target_loop_class = self.bootstrap_data.selected_loop_classes[idx_bootstrap][
            target_class_idx
        ]
        if (
            source_loop_class is None
            or source_loop_class.representatives is None
            or target_loop_class is None
            or target_loop_class.representatives is None
        ):
            return (source_class_idx, target_class_idx, False)

        source_loops = source_loop_class.representatives
        target_loops = target_loop_class.representatives
        if len(source_loops) == 0 or len(target_loops) == 0:
            return (source_class_idx, target_class_idx, False)

        source_loop_death = source_loop_class.death

        mask_a = self._loops_to_edge_mask(source_loops)
        mask_b = self._loops_to_edge_mask(target_loops)

        boundary_matrix_d1 = self.boundary_matrix_d1
        assert boundary_matrix_d1 is not None
        results, _ = compute_loop_homological_equivalence(
            boundary_matrix_d1=boundary_matrix_d1,
            loop_mask_a=mask_a,
            loop_mask_b=mask_b,
            n_pairs_check=n_pairs_check,
            max_column_diameter=source_loop_death,
        )
        return (source_class_idx, target_class_idx, any(r == 0 for r in results))

    def _ensure_loop_tracks(self) -> None:
        if self.bootstrap_data is None:
            return
        if self.persistence_diagram is None:
            return
        top_k = len(self.selected_loop_classes)
        if top_k == 0:
            return
        for idx_track in range(top_k):
            if idx_track not in self.bootstrap_data.loop_tracks:
                self.bootstrap_data.loop_tracks[idx_track] = LoopTrack(
                    source_class_idx=idx_track
                )

    def _bootstrap(
        self,
        adata: AnnData,
        n_bootstrap: Count_t,
        thresh: Diameter_t | None = None,
        top_k: int = 1,
        noise_scale: float = 1e-3,
        n_reps_per_loop: int = 4,
        life_pct: float = 0.1,
        n_cocycles_used: int = 3,
        n_force_deviate: int = 4,
        k_yen: int = 8,
        loop_lower_t_pct: float = 2.5,
        loop_upper_t_pct: float = 97.5,
        n_pairs_check_equivalence: int = 4,
        n_max_workers: int = DEFAULT_N_MAX_WORKERS,
        k_neighbors_check_equivalence: int = 3,
        method_geometric_equivalence: LoopDistMethod = "hausdorff",
        verbose: bool = False,
        **nei_kwargs,
    ) -> None:
        self.bootstrap_data = BootstrapAnalysis()
        if self.meta.bootstrap is None:
            self.meta.bootstrap = BootstrapMeta(indices_resample=[])
        else:
            if self.meta.bootstrap.indices_resample is None:
                self.meta.bootstrap.indices_resample = []
            else:
                self.meta.bootstrap.indices_resample.clear()

        console = Console()
        progress_main = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        logger.remove()
        # https://github.com/Delgan/loguru/issues/444
        logger.add(
            lambda s: console.print(s, end=""),
            colorize=False,
            level="TRACE",
            format="<green>{time:YYYY/MM/DD HH:mm:ss}</green> | {level.icon} - <level>{message}</level>",
        )

        with progress_main:
            for idx_bootstrap in progress_main.track(range(n_bootstrap)):
                start_time = time.perf_counter()
                if verbose:
                    logger.info(f"Start round {idx_bootstrap + 1}/{n_bootstrap}")
                if verbose:
                    logger.info("Computing bootstrapped homology")
                pairwise_distance_matrix = self._compute_homology(
                    adata=adata,
                    thresh=thresh,
                    bootstrap=True,
                    noise_scale=noise_scale,
                    **nei_kwargs,
                )
                if verbose:
                    logger.info("Finding loops in the bootstrapped data")
                assert self.meta.preprocess is not None
                assert self.meta.preprocess.embedding_method is not None
                embedding = np.array(
                    adata.obsm[f"X_{self.meta.preprocess.embedding_method}"]
                )
                self._compute_loop_representatives(
                    embedding=embedding,
                    pairwise_distance_matrix=pairwise_distance_matrix,
                    idx_bootstrap=idx_bootstrap,
                    top_k=top_k,
                    bootstrap=True,
                    n_reps_per_loop=n_reps_per_loop,
                    life_pct=life_pct,
                    n_cocycles_used=n_cocycles_used,
                    n_force_deviate=n_force_deviate,
                    k_yen=k_yen,
                    loop_lower_t_pct=loop_lower_t_pct,
                    loop_upper_t_pct=loop_upper_t_pct,
                )
                if verbose:
                    logger.info("Matching bootstrapped loops to the original loops")
                """
                ============= geometric matching =============
                - find loop neighbors using hausdorff/frechet
                - reduce computation load
                ==============================================
                """
                n_original_loop_classes = len(self.selected_loop_classes)
                n_bootstrap_loop_classes = len(
                    self.bootstrap_data.selected_loop_classes[idx_bootstrap]
                )

                if n_original_loop_classes == 0 or n_bootstrap_loop_classes == 0:
                    continue

                logger.info(
                    f"[Bootstrap {idx_bootstrap + 1}/{n_bootstrap}] "
                    f"Geometric matching candidates: {n_original_loop_classes} original classes "
                    f"x {n_bootstrap_loop_classes} bootstrap classes"
                )

                pairwise_result_matrix = np.full(
                    (n_original_loop_classes, n_bootstrap_loop_classes), np.nan
                )

                with ThreadPoolExecutor(max_workers=n_max_workers) as executor:
                    tasks = {}
                    for i in range(n_original_loop_classes):
                        for j in range(n_bootstrap_loop_classes):
                            task = executor.submit(
                                self._assess_bootstrap_geometric_equivalence,
                                source_class_idx=i,
                                target_class_idx=j,
                                idx_bootstrap=idx_bootstrap,
                                method=method_geometric_equivalence,
                            )
                            tasks[task] = (i, j)

                    for task in as_completed(tasks):
                        src_idx, tgt_idx, distance = task.result()
                        pairwise_result_matrix[src_idx, tgt_idx] = distance

                neighbor_indices, neighbor_distances = nearest_neighbor_per_row(
                    pairwise_result_matrix, k_neighbors_check_equivalence
                )

                logger.info(
                    f"[Bootstrap {idx_bootstrap + 1}/{n_bootstrap}] "
                    f"Geometric neighbors chosen per class: {k_neighbors_check_equivalence}"
                )
                """
                ========= homological matching =========
                - gf2 regression
                ========================================
                """
                with ThreadPoolExecutor(max_workers=n_max_workers) as executor:
                    tasks = {}
                    for si in range(n_original_loop_classes):
                        for k in range(k_neighbors_check_equivalence):
                            tj = neighbor_indices[si, k]
                            if tj >= 0:
                                task = executor.submit(
                                    self._assess_bootstrap_homology_equivalence,
                                    source_class_idx=si,
                                    target_class_idx=tj,
                                    idx_bootstrap=idx_bootstrap,
                                    n_pairs_check=n_pairs_check_equivalence,
                                )
                                tasks[task] = (si, tj, neighbor_distances[si, k], k)

                    for task in as_completed(tasks):
                        si, tj, geo_dist, neighbor_rank = tasks[task]
                        _, _, is_homologically_equivalent = task.result()
                        if (
                            self.bootstrap_data is not None
                            and is_homologically_equivalent
                        ):
                            logger.info(
                                f"[Bootstrap {idx_bootstrap + 1}/{n_bootstrap}] "
                                f"Homology match found: original class #{si}↔bootstrap class #{tj} "
                                f"({method_geometric_equivalence} distance={geo_dist:.4f}, "
                                f"neighbor rank {neighbor_rank + 1}/{k_neighbors_check_equivalence})"
                            )
                            self._ensure_loop_tracks()
                            track: LoopTrack = self.bootstrap_data.loop_tracks[si]
                            track.matches.append(
                                LoopMatch(
                                    idx_bootstrap=idx_bootstrap,
                                    target_class_idx=tj,
                                    geometric_distance=float(geo_dist),
                                    neighbor_rank=neighbor_rank,
                                )
                            )
                self.bootstrap_data.num_bootstraps += 1
                end_time = time.perf_counter()
                if verbose:
                    time_elapsed = end_time - start_time
                    logger.success(
                        f"Round {idx_bootstrap + 1}/{n_bootstrap} finished in {int(time_elapsed // 3600)}h {int(time_elapsed % 3600 // 60)}m {int(time_elapsed % 60)}s"
                    )

    def _test_loops(
        self,
        method_pval_correction: MultipleTestCorrectionMethod = "benjamini-hochberg",
    ) -> None:
        if self.bootstrap_data is None:
            return
        if self.bootstrap_data.num_bootstraps == 0:
            return

        self.bootstrap_data.fisher_presence_results = (
            self.bootstrap_data.fisher_test_presence(
                method_pval_correction=method_pval_correction
            )
        )

        self.bootstrap_data.gamma_persistence_results = (
            self.bootstrap_data.gamma_test_persistence(
                self.selected_loop_classes, method_pval_correction
            )
        )
