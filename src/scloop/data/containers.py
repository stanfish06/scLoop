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
from scipy.spatial.distance import directed_hausdorff

from ..computing.homology import (
    compute_boundary_matrix_data,
    compute_loop_homological_equivalence,
    compute_persistence_diagram_and_cocycles,
)
from .analysis_containers import (
    BootstrapAnalysis,
    HodgeAnalysis,
    LoopMatch,
    LoopTrack,
)
from .loop_reconstruction import reconstruct_n_loop_representatives
from .metadata import BootstrapMeta, ScloopMeta
from .types import Diameter_t, Index_t, IndexListDownSample, LoopDistMethod, Size_t
from .utils import (
    decode_edges,
    decode_triangles,
    extract_edges_from_coo,
    loop_vertices_to_edge_ids,
    nearest_neighbor_per_row,
)


class BoundaryMatrix(BaseModel, ABC):
    num_vertices: Size_t
    data: tuple[list, list]  # in coo format (row indices, col indices) of ones
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
    data: tuple[list[Index_t], list[Index_t]]

    def row_simplex_decode(self) -> list[tuple[Index_t, Index_t]]:
        return decode_edges(np.array(self.row_simplex_ids), self.num_vertices)

    def col_simplex_decode(self) -> list[tuple[Index_t, Index_t, Index_t]]:
        return decode_triangles(np.array(self.col_simplex_ids), self.num_vertices)


@dataclass
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

    def _loops_to_edge_mask(self, loops: list[list[int]]) -> np.ndarray:
        assert self.boundary_matrix_d1 is not None
        num_vertices = self.boundary_matrix_d1.num_vertices
        n_edges = self.boundary_matrix_d1.shape[0]

        edge_lookup = {
            int(edge_id): row_idx
            for row_idx, edge_id in enumerate(self.boundary_matrix_d1.row_simplex_ids)
        }

        mask = np.zeros((len(loops), n_edges), dtype=bool)
        for idx, loop in enumerate(loops):
            edge_ids = loop_vertices_to_edge_ids(
                np.asarray(loop, dtype=np.int64), num_vertices
            )
            row_ids = [edge_lookup.get(int(eid), -1) for eid in edge_ids]
            row_ids = [rid for rid in row_ids if rid >= 0]
            if row_ids:
                mask[idx, row_ids] = True
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

    def _compute_boundary_matrix(
        self, adata: AnnData, thresh: Diameter_t | None = None, **nei_kwargs
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
        self.boundary_matrix_d1 = BoundaryMatrixD1(
            num_vertices=self.meta.preprocess.num_vertices,
            data=(
                edge_ids_reindex.flatten().tolist(),
                np.repeat(
                    np.expand_dims(np.arange(edge_ids_reindex.shape[0]), 1), 3, axis=1
                )
                .flatten()
                .tolist(),
            ),
            shape=(len(edge_ids_1d), len(trig_ids)),
            row_simplex_ids=edge_ids_1d.tolist(),
            col_simplex_ids=trig_ids,
            row_simplex_diams=row_simplex_diams.tolist(),
            col_simplex_diams=result.triangle_diameters,
        )

    # ISSUE: currently, cocycles and loop representatives are de-coupled (for the ease of checking matches for bootstrap)
    def _compute_loop_representatives(
        self,
        pairwise_distance_matrix: csr_matrix,
        top_k: int | None = None,  # top k homology classes to compute representatives
        bootstrap: bool = False,
        idx_bootstrap: int = 0,
        n_reps_per_loop: int = 4,
        life_pct: float = 0.1,
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
            if self.meta.preprocess.indices_downsample is not None:
                vertex_ids: IndexListDownSample = (
                    self.meta.preprocess.indices_downsample
                )
            else:
                vertex_ids = (
                    np.arange(pairwise_distance_matrix.shape[0])
                    .astype(np.int64)
                    .tolist()
                )
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
            vertex_ids: IndexListDownSample = self.meta.bootstrap.indices_resample[
                idx_bootstrap
            ]

        if loop_births.size == 0:
            return [], []
        if top_k is None:
            top_k = loop_births.size
        if top_k <= 0:
            return [], []
        top_k = min(top_k, loop_births.size)

        indices_top_k = np.argpartition(loop_deaths - loop_births, -top_k)[-top_k:]

        dm_upper = triu(pairwise_distance_matrix, k=1).tocoo()
        edges_array, edge_diameters = extract_edges_from_coo(
            dm_upper.row, dm_upper.col, dm_upper.data
        )

        if len(edges_array) == 0:
            return [], []

        if not bootstrap:
            if self.loop_representatives is None:
                self.loop_representatives = []
            while len(self.loop_representatives) < len(indices_top_k):
                self.loop_representatives.append([])
        else:
            assert self.bootstrap_data is not None
            bootstrap_data = self.bootstrap_data
            while len(bootstrap_data.loop_representatives) <= idx_bootstrap:
                bootstrap_data.loop_representatives.append([])
            if len(bootstrap_data.loop_representatives[idx_bootstrap]) < len(
                indices_top_k
            ):
                bootstrap_data.loop_representatives[idx_bootstrap] = [
                    [] for _ in range(len(indices_top_k))
                ]

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
                assert self.bootstrap_data is not None
                self.bootstrap_data.loop_representatives[idx_bootstrap][loop_idx] = (
                    loops
                )

    def _assess_bootstrap_geometric_equivalence(
        self,
        adata: AnnData,
        source_class_idx: int,
        target_class_idx: int,
        idx_bootstrap: int = 0,
        method: LoopDistMethod = "hausdorff",
    ) -> tuple[int, int, float]:
        assert self.loop_representatives is not None
        assert self.bootstrap_data is not None
        assert self.meta.preprocess is not None
        assert self.meta.preprocess.embedding_method is not None

        if idx_bootstrap >= len(self.bootstrap_data.loop_representatives):
            return (source_class_idx, target_class_idx, np.nan)
        if source_class_idx >= len(self.loop_representatives):
            return (source_class_idx, target_class_idx, np.nan)

        boot_loops_all = self.bootstrap_data.loop_representatives[idx_bootstrap]
        if target_class_idx >= len(boot_loops_all):
            return (source_class_idx, target_class_idx, np.nan)

        source_loops = self.loop_representatives[source_class_idx]
        target_loops = boot_loops_all[target_class_idx]

        if len(source_loops) == 0 or len(target_loops) == 0:
            return (source_class_idx, target_class_idx, np.nan)

        emb = adata.obsm[f"X_{self.meta.preprocess.embedding_method}"]
        distances = []
        for source_loop in source_loops:
            for target_loop in target_loops:
                source_coords = emb[source_loop]
                target_coords = emb[target_loop]
                try:
                    dist = max(
                        directed_hausdorff(source_coords, target_coords)[0],
                        directed_hausdorff(target_coords, source_coords)[0],
                    )
                    distances.append(dist)
                except (ValueError, IndexError):
                    distances.append(np.nan)

        mean_distance = np.nanmean(distances) if distances else np.nan
        return (source_class_idx, target_class_idx, float(mean_distance))

    def _assess_bootstrap_homology_equivalence(
        self,
        source_class_idx: int,
        target_class_idx: int | None = None,
        idx_bootstrap: int = 0,
        n_pairs_check: int = 10,
    ) -> tuple[int, int, bool]:
        assert self.boundary_matrix_d1 is not None
        assert self.loop_representatives is not None
        assert self.bootstrap_data is not None
        if target_class_idx is None:
            target_class_idx = source_class_idx
        if idx_bootstrap >= len(self.bootstrap_data.loop_representatives):
            return (source_class_idx, target_class_idx, False)
        if source_class_idx >= len(self.loop_representatives):
            return (source_class_idx, target_class_idx, False)
        boot_loops_all = self.bootstrap_data.loop_representatives[idx_bootstrap]
        if target_class_idx >= len(boot_loops_all):
            return (source_class_idx, target_class_idx, False)

        source_loops = self.loop_representatives[source_class_idx]
        target_loops = boot_loops_all[target_class_idx]
        if len(source_loops) == 0 or len(target_loops) == 0:
            return (source_class_idx, target_class_idx, False)

        mask_a = self._loops_to_edge_mask(source_loops)
        mask_b = self._loops_to_edge_mask(target_loops)

        results, _ = compute_loop_homological_equivalence(
            boundary_matrix_d1=self.boundary_matrix_d1,
            loop_mask_a=mask_a,
            loop_mask_b=mask_b,
            n_pairs_check=n_pairs_check,
        )
        return (source_class_idx, target_class_idx, any(r == 0 for r in results))

    def _ensure_loop_tracks(self) -> None:
        if self.bootstrap_data is None:
            return
        if self.persistence_diagram is None or self.loop_representatives is None:
            return
        loop_births = np.array(self.persistence_diagram[1][0], dtype=np.float32)
        loop_deaths = np.array(self.persistence_diagram[1][1], dtype=np.float32)
        top_k = len(self.loop_representatives)
        if top_k == 0:
            return
        indices_top_k = np.argpartition(loop_deaths - loop_births, -top_k)[-top_k:]
        for track_idx, loop_idx in enumerate(indices_top_k):
            birth = float(loop_births[loop_idx])
            death = float(loop_deaths[loop_idx])
            if track_idx not in self.bootstrap_data.loop_tracks:
                self.bootstrap_data.loop_tracks[track_idx] = LoopTrack(
                    source_class_idx=track_idx, birth_root=birth, death_root=death
                )

    def _bootstrap(
        self,
        adata: AnnData,
        n_bootstrap: int,
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
        n_max_workers: int = 8,
        k_neighbors_check_equivalence: int = 3,
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
                    logger.info("Compute bootstrapped homology")
                pairwise_distance_matrix = self._compute_homology(
                    adata=adata,
                    thresh=thresh,
                    bootstrap=True,
                    noise_scale=noise_scale,
                    **nei_kwargs,
                )
                if verbose:
                    logger.info("Find loops in the bootstrapped data")
                self._compute_loop_representatives(
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
                    logger.info("Match bootstrapped loops to the original loops")
                """
                ============= geometric matching =============
                - find loop neighbors using hausdorff/frechet
                - reduce computation load
                ==============================================
                """
                n_original_loop_classes = len(self.loop_representatives)
                n_bootstrap_loop_classes = len(
                    self.bootstrap_data.loop_representatives[idx_bootstrap]
                )

                if n_original_loop_classes == 0 or n_bootstrap_loop_classes == 0:
                    continue

                pairwise_result_matrix = np.full(
                    (n_original_loop_classes, n_bootstrap_loop_classes), np.nan
                )

                with ThreadPoolExecutor(max_workers=n_max_workers) as executor:
                    tasks = {}
                    for i in range(n_original_loop_classes):
                        for j in range(n_bootstrap_loop_classes):
                            task = executor.submit(
                                self._assess_bootstrap_geometric_equivalence,
                                adata,
                                i,
                                j,
                                idx_bootstrap,
                            )
                            tasks[task] = (i, j)

                    for task in as_completed(tasks):
                        src_idx, tgt_idx, distance = task.result()
                        pairwise_result_matrix[src_idx, tgt_idx] = distance

                neighbor_indices, neighbor_distances = nearest_neighbor_per_row(
                    pairwise_result_matrix, k_neighbors_check_equivalence
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
                                    si,
                                    tj,
                                    idx_bootstrap,
                                    n_pairs_check_equivalence,
                                )
                                tasks[task] = (si, tj, neighbor_distances[si, k], k)

                    for task in as_completed(tasks):
                        si, tj, geo_dist, neighbor_rank = tasks[task]
                        _, _, is_homologically_equivalent = task.result()
                        if (
                            self.bootstrap_data is not None
                            and is_homologically_equivalent
                        ):
                            self._ensure_loop_tracks()
                            track: LoopTrack = self.bootstrap_data.loop_tracks[si]
                            birth_boot = float(
                                self.bootstrap_data.persistence_diagrams[idx_bootstrap][
                                    1
                                ][0][tj]
                            )
                            death_boot = float(
                                self.bootstrap_data.persistence_diagrams[idx_bootstrap][
                                    1
                                ][1][tj]
                            )
                            track.matches.append(
                                LoopMatch(
                                    idx_bootstrap=idx_bootstrap,
                                    birth_bootstrap=birth_boot,
                                    death_bootstrap=death_boot,
                                    target_class_idx=tj,
                                    geometric_distance=float(geo_dist),
                                    neighbor_rank=neighbor_rank,
                                )
                            )
                self.bootstrap_data.num_bootstraps += 1
                end_time = time.perf_counter()
                if verbose:
                    time_elapsed = end_time - start_time
                    console.clear()
                    logger.success(
                        f"Round {idx_bootstrap + 1}/{n_bootstrap} finished in {int(time_elapsed // 3600)}h {int(time_elapsed % 3600 // 60)}m {int(time_elapsed % 60)}s"
                    )
