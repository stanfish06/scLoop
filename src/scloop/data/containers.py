# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from anndata import AnnData
from loguru import logger
from pydantic import Field
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
from scipy.sparse import csr_matrix

from ..analyzing.bootstrap import run_bootstrap_pipeline
from ..analyzing.hodge import compute_hodge_analysis
from ..computing.boundary import (
    compute_boundary_matrix_d0,
    compute_boundary_matrix_d1,
)
from ..computing.hodge_decomposition import (
    compute_hodge_eigendecomposition,
    compute_hodge_matrix,
)
from ..computing.homology import (
    compute_persistence_diagram_and_cocycles,
)
from ..computing.loops import compute_loop_representatives
from ..computing.matching import (
    check_homological_equivalence,
    compute_geometric_distance,
    loops_to_edge_mask,
)
from .analysis_containers import (
    BootstrapAnalysis,
    LoopMatch,
    LoopTrack,
)
from .base_components import LoopClass
from .boundary import BoundaryMatrixD0, BoundaryMatrixD1
from .constants import (
    DEFAULT_EXTRA_DIAM_EQUIVALENCE,
    DEFAULT_HALF_WINDOW,
    DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
    DEFAULT_K_YEN,
    DEFAULT_LIFE_PCT,
    DEFAULT_LOOP_DIST_METHOD,
    DEFAULT_MAXITER_EIGENDECOMPOSITION,
    DEFAULT_N_COCYCLES_USED,
    DEFAULT_N_FORCE_DEVIATE,
    DEFAULT_N_HODGE_COMPONENTS,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    DEFAULT_N_PAIRS_CHECK,
    DEFAULT_N_PAIRS_CHECK_EQUIVALENCE,
    DEFAULT_N_REPS_PER_LOOP,
    DEFAULT_NOISE_SCALE,
    DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    DEFAULT_WEIGHT_HODGE,
)
from .metadata import BootstrapMeta, ScloopMeta
from .types import (
    Count_t,
    Diameter_t,
    Index_t,
    IndexListDownSample,
    LoopDistMethod,
    MultipleTestCorrectionMethod,
    Percent_t,
    PositiveFloat,
)
from .utils import (
    loops_to_coords,
    nearest_neighbor_per_row,
)


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
        return loops_to_edge_mask(
            loops=loops,
            boundary_matrix_d1=self.boundary_matrix_d1,
            return_valid_indices=return_valid_indices,
            use_order=use_order,
        )

    def to_hdf5_group(self, group, compress: bool = True) -> None:
        kw = {"compression": "gzip"} if compress else {}

        # meta
        meta_grp = group.create_group("meta")
        self.meta.to_hdf5_group(meta_grp, compress=compress)

        # persistence_diagram: list of [births, deaths] per dimension
        if self.persistence_diagram is not None:
            pd_grp = group.create_group("persistence_diagram")
            pd_grp.attrs["_n_dims"] = len(self.persistence_diagram)
            for dim_idx, dim_pd in enumerate(self.persistence_diagram):
                dim_grp = pd_grp.create_group(str(dim_idx))
                if dim_pd is not None and len(dim_pd) >= 2:
                    dim_grp.create_dataset(
                        "births", data=np.asarray(dim_pd[0], dtype=np.float64), **kw
                    )
                    dim_grp.create_dataset(
                        "deaths", data=np.asarray(dim_pd[1], dtype=np.float64), **kw
                    )

        # cocycles: list of list of (vertices, coeff) per dimension
        if self.cocycles is not None:
            cc_grp = group.create_group("cocycles")
            cc_grp.attrs["_n_dims"] = len(self.cocycles)
            for dim_idx, dim_cocycles in enumerate(self.cocycles):
                dim_grp = cc_grp.create_group(str(dim_idx))
                if dim_cocycles is not None:
                    dim_grp.attrs["_n_cocycles"] = len(dim_cocycles)
                    for cc_idx, cocycle in enumerate(dim_cocycles):
                        cc_subgrp = dim_grp.create_group(str(cc_idx))
                        if cocycle is not None and len(cocycle) > 0:
                            verts_list = []
                            coeffs_list = []
                            for simplex in cocycle:
                                try:
                                    verts, coeff = simplex
                                    verts_list.append(list(verts))
                                    coeffs_list.append(int(coeff))
                                except (ValueError, TypeError):
                                    continue
                            if verts_list:
                                max_len = max(len(v) for v in verts_list)
                                verts_arr = np.full(
                                    (len(verts_list), max_len), -1, dtype=np.int64
                                )
                                for i, v in enumerate(verts_list):
                                    verts_arr[i, : len(v)] = v
                                cc_subgrp.create_dataset(
                                    "vertices", data=verts_arr, **kw
                                )
                                cc_subgrp.create_dataset(
                                    "coefficients",
                                    data=np.array(coeffs_list, dtype=np.int32),
                                    **kw,
                                )

        # selected_loop_classes
        slc_grp = group.create_group("selected_loop_classes")
        slc_grp.attrs["_count"] = len(self.selected_loop_classes)
        for i, lc in enumerate(self.selected_loop_classes):
            lc_grp = slc_grp.create_group(str(i))
            if lc is None:
                lc_grp.attrs["_is_none"] = True
            else:
                lc_grp.attrs["_is_none"] = False
                lc.to_hdf5_group(lc_grp, compress=compress)

        # boundary matrices
        if self.boundary_matrix_d1 is not None:
            bd1_grp = group.create_group("boundary_matrix_d1")
            self.boundary_matrix_d1.to_hdf5_group(bd1_grp, compress=compress)

        if self.boundary_matrix_d0 is not None:
            bd0_grp = group.create_group("boundary_matrix_d0")
            self.boundary_matrix_d0.to_hdf5_group(bd0_grp, compress=compress)

        # bootstrap data
        if self.bootstrap_data is not None:
            boot_grp = group.create_group("bootstrap_data")
            self.bootstrap_data.to_hdf5_group(boot_grp, compress=compress)

    @classmethod
    def from_hdf5_group(cls, group) -> "HomologyData":
        import h5py

        # meta
        meta_grp: h5py.Group = group["meta"]  # type: ignore[assignment]
        meta = ScloopMeta.from_hdf5_group(meta_grp)

        # persistence_diagram
        persistence_diagram = None
        if "persistence_diagram" in group:
            pd_grp: h5py.Group = group["persistence_diagram"]  # type: ignore[assignment]
            n_dims = int(pd_grp.attrs["_n_dims"])  # type: ignore[arg-type]
            persistence_diagram = []
            for dim_idx in range(n_dims):
                dim_grp: h5py.Group = pd_grp[str(dim_idx)]  # type: ignore[assignment]
                if "births" in dim_grp and "deaths" in dim_grp:
                    births = np.asarray(dim_grp["births"]).tolist()
                    deaths = np.asarray(dim_grp["deaths"]).tolist()
                    persistence_diagram.append([births, deaths])
                else:
                    persistence_diagram.append(None)

        # cocycles
        cocycles = None
        if "cocycles" in group:
            cc_grp: h5py.Group = group["cocycles"]  # type: ignore[assignment]
            n_dims = int(cc_grp.attrs["_n_dims"])  # type: ignore[arg-type]
            cocycles = []
            for dim_idx in range(n_dims):
                dim_grp: h5py.Group = cc_grp[str(dim_idx)]  # type: ignore[assignment]
                n_cocycles = int(dim_grp.attrs.get("_n_cocycles", 0))  # type: ignore[arg-type]
                dim_cocycles = []
                for cc_idx in range(n_cocycles):
                    cc_subgrp: h5py.Group = dim_grp[str(cc_idx)]  # type: ignore[assignment]
                    if "vertices" in cc_subgrp and "coefficients" in cc_subgrp:
                        verts_arr = np.asarray(cc_subgrp["vertices"])
                        coeffs_arr = np.asarray(cc_subgrp["coefficients"])
                        cocycle = []
                        for i in range(len(coeffs_arr)):
                            verts = [int(v) for v in verts_arr[i] if v >= 0]
                            cocycle.append((verts, int(coeffs_arr[i])))
                        dim_cocycles.append(cocycle)
                    else:
                        dim_cocycles.append([])
                cocycles.append(dim_cocycles)

        # selected_loop_classes
        selected_loop_classes: list[LoopClass | None] = []
        slc_grp: h5py.Group = group["selected_loop_classes"]  # type: ignore[assignment]
        n_lcs = int(slc_grp.attrs["_count"])  # type: ignore[arg-type]
        for i in range(n_lcs):
            lc_grp: h5py.Group = slc_grp[str(i)]  # type: ignore[assignment]
            if lc_grp.attrs.get("_is_none", False):
                selected_loop_classes.append(None)
            else:
                selected_loop_classes.append(LoopClass.from_hdf5_group(lc_grp))

        # boundary matrices
        boundary_matrix_d1 = None
        if "boundary_matrix_d1" in group:
            bd1_grp: h5py.Group = group["boundary_matrix_d1"]  # type: ignore[assignment]
            boundary_matrix_d1 = BoundaryMatrixD1.from_hdf5_group(bd1_grp)

        boundary_matrix_d0 = None
        if "boundary_matrix_d0" in group:
            bd0_grp: h5py.Group = group["boundary_matrix_d0"]  # type: ignore[assignment]
            boundary_matrix_d0 = BoundaryMatrixD0.from_hdf5_group(bd0_grp)

        # bootstrap data
        bootstrap_data = None
        if "bootstrap_data" in group:
            boot_grp: h5py.Group = group["bootstrap_data"]  # type: ignore[assignment]
            bootstrap_data = BootstrapAnalysis.from_hdf5_group(boot_grp)

        return cls(
            meta=meta,
            persistence_diagram=persistence_diagram,
            cocycles=cocycles,
            selected_loop_classes=selected_loop_classes,
            boundary_matrix_d1=boundary_matrix_d1,
            boundary_matrix_d0=boundary_matrix_d0,
            bootstrap_data=bootstrap_data,
        )

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
        self.boundary_matrix_d1 = compute_boundary_matrix_d1(
            adata=adata,
            meta=self.meta,
            thresh=thresh,
            verbose=verbose,
            **nei_kwargs,
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

        vertex_ids = sorted(self._original_vertex_ids)
        self.boundary_matrix_d0 = compute_boundary_matrix_d0(
            boundary_matrix_d1=self.boundary_matrix_d1,
            num_vertices=self.meta.preprocess.num_vertices,
            vertex_ids=vertex_ids,
            verbose=verbose,
        )

    def _compute_hodge_matrix(
        self, thresh: Diameter_t, normalized: bool = True
    ) -> csr_matrix | None:
        if self.boundary_matrix_d0 is None or self.boundary_matrix_d1 is None:
            raise ValueError("Boundary matrices must be computed first.")

        d1_rows, d1_cols, d1_vals = self.boundary_matrix_d0.data
        bd0 = csr_matrix(
            (d1_vals, (d1_rows, d1_cols)), shape=self.boundary_matrix_d0.shape
        )

        d2_rows, d2_cols, d2_vals = self.boundary_matrix_d1.data
        bd1 = csr_matrix(
            (d2_vals, (d2_rows, d2_cols)), shape=self.boundary_matrix_d1.shape
        )

        triangle_diams = np.array(self.boundary_matrix_d1.col_simplex_diams)

        return compute_hodge_matrix(
            boundary_matrix_d0=bd0,
            boundary_matrix_d1=bd1,
            triangle_diams=triangle_diams,
            thresh=thresh,
            normalized=normalized,
        )

    def _compute_hodge_eigendecomposition(
        self,
        hodge_matrix: csr_matrix,
        timeout: float = DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
        n_components: int = DEFAULT_N_HODGE_COMPONENTS,
        maxiter: int | None = DEFAULT_MAXITER_EIGENDECOMPOSITION,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return compute_hodge_eigendecomposition(
            hodge_matrix=hodge_matrix,
            n_components=n_components,
            timeout=timeout,
            maxiter=maxiter,
        )

    def _compute_hodge_analysis_for_track(
        self,
        idx_track: Index_t,  # TODO: potentially allow multiple tracks and parallelize them
        values_vertices: np.ndarray,
        coordinates_vertices: np.ndarray | None = None,
        life_pct: Percent_t | None = None,
        n_hodge_components: int = DEFAULT_N_HODGE_COMPONENTS,
        normalized: bool = True,
        n_neighbors_edge_embedding: Count_t = DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
        compute_gene_trends: bool = True,
        gene_expression_matrix: Any | None = None,
        gene_names: list[str] | None = None,
        verbose: bool = False,
        progress: Progress | None = None,
        timeout_eigendecomposition: float = DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
        maxiter_eigendecomposition: int | None = DEFAULT_MAXITER_EIGENDECOMPOSITION,
        kwargs_edge_embedding: dict[str, Any] | None = None,
        kwargs_trajectory: dict[str, Any] | None = None,
        kwargs_gene_trends: dict[str, Any] | None = None,
    ) -> None:
        assert self.bootstrap_data is not None
        assert self.boundary_matrix_d0 is not None
        assert self.boundary_matrix_d1 is not None

        kwargs_edge_embedding = kwargs_edge_embedding or {}
        kwargs_trajectory = kwargs_trajectory or {}
        kwargs_gene_trends = kwargs_gene_trends or {}

        weight_hodge = kwargs_edge_embedding.get("weight_hodge", DEFAULT_WEIGHT_HODGE)
        half_window = kwargs_edge_embedding.get("half_window", DEFAULT_HALF_WINDOW)
        gene_trend_confidence_level = kwargs_gene_trends.get("confidence_level", 0.95)

        compute_hodge_analysis(
            idx_track=idx_track,
            track_id=idx_track,
            bootstrap_data=self.bootstrap_data,
            selected_loop_classes=self.selected_loop_classes,
            boundary_matrix_d0=self.boundary_matrix_d0,
            boundary_matrix_d1=self.boundary_matrix_d1,
            meta=self.meta,
            values_vertices=values_vertices,
            coordinates_vertices=coordinates_vertices,
            life_pct=life_pct,
            n_hodge_components=n_hodge_components,
            normalized=normalized,
            n_neighbors_edge_embedding=n_neighbors_edge_embedding,
            weight_hodge=weight_hodge,
            half_window=half_window,
            compute_gene_trends=compute_gene_trends,
            gene_expression_matrix=gene_expression_matrix,
            gene_names=gene_names,
            gene_trend_confidence_level=gene_trend_confidence_level,
            verbose=verbose,
            progress=progress,
            timeout_eigendecomposition=timeout_eigendecomposition,
            maxiter_eigendecomposition=maxiter_eigendecomposition,
            kwargs_trajectory=kwargs_trajectory,
        )

    def _compute_loop_representatives(
        self,
        embedding: np.ndarray,
        pairwise_distance_matrix: csr_matrix,
        top_k: int | None = None,
        bootstrap: bool = False,
        idx_bootstrap: Index_t = 0,
        n_reps_per_loop: Count_t = DEFAULT_N_REPS_PER_LOOP,
        life_pct: Percent_t = DEFAULT_LIFE_PCT,
        n_cocycles_used: Count_t = DEFAULT_N_COCYCLES_USED,
        n_force_deviate: Count_t = DEFAULT_N_FORCE_DEVIATE,
        k_yen: Count_t = DEFAULT_K_YEN,
        loop_lower_t_pct: float = 2.5,
        loop_upper_t_pct: float = 97.5,
    ):
        assert pairwise_distance_matrix.shape is not None
        assert self.meta.preprocess is not None
        assert self.boundary_matrix_d1 is not None

        # Extract data from self based on bootstrap flag
        if not bootstrap:
            assert self.persistence_diagram is not None
            assert self.cocycles is not None
            persistence_diagram = self.persistence_diagram[1]
            cocycles = self.cocycles[1]
            vertex_ids = self._original_vertex_ids
        else:
            assert self.bootstrap_data is not None
            assert len(self.bootstrap_data.persistence_diagrams) > idx_bootstrap
            assert len(self.bootstrap_data.cocycles) > idx_bootstrap
            assert self.meta.bootstrap is not None
            assert self.meta.bootstrap.indices_resample is not None
            assert len(self.meta.bootstrap.indices_resample) > idx_bootstrap
            persistence_diagram = self.bootstrap_data.persistence_diagrams[
                idx_bootstrap
            ][1]
            cocycles = self.bootstrap_data.cocycles[idx_bootstrap][1]
            vertex_ids = self.meta.bootstrap.indices_resample[idx_bootstrap]

        loop_classes = compute_loop_representatives(
            embedding=embedding,
            pairwise_distance_matrix=pairwise_distance_matrix,
            persistence_diagram=persistence_diagram,
            cocycles=cocycles,
            boundary_matrix_d1=self.boundary_matrix_d1,
            vertex_ids=vertex_ids,
            top_k=top_k,
            n_reps_per_loop=n_reps_per_loop,
            life_pct=life_pct,
            n_cocycles_used=n_cocycles_used,
            n_force_deviate=n_force_deviate,
            k_yen=k_yen,
            loop_lower_t_pct=loop_lower_t_pct,
            loop_upper_t_pct=loop_upper_t_pct,
            bootstrap=bootstrap,
            rank_offset=0,
        )

        if not bootstrap:
            while len(self.selected_loop_classes) < len(loop_classes):
                self.selected_loop_classes.append(None)
            for i, loop_class in enumerate(loop_classes):
                self.selected_loop_classes[i] = loop_class
        else:
            assert self.bootstrap_data is not None
            while len(self.bootstrap_data.selected_loop_classes) <= idx_bootstrap:
                self.bootstrap_data.selected_loop_classes.append([])
            while len(self.bootstrap_data.selected_loop_classes[idx_bootstrap]) < len(
                loop_classes
            ):
                self.bootstrap_data.selected_loop_classes[idx_bootstrap].append(None)
            for i, loop_class in enumerate(loop_classes):
                self.bootstrap_data.selected_loop_classes[idx_bootstrap][i] = loop_class

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
        method: LoopDistMethod = DEFAULT_LOOP_DIST_METHOD,
        n_workers: Count_t = DEFAULT_N_MAX_WORKERS,
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

        mean_distance = compute_geometric_distance(
            source_coords_list=source_coords_list,
            target_coords_list=target_coords_list,
            method=method,
            n_workers=n_workers,
        )
        return (source_class_idx, target_class_idx, mean_distance)

    def _assess_bootstrap_homology_equivalence(
        self,
        source_class_idx: Index_t,
        target_class_idx: Index_t | None = None,
        idx_bootstrap: Index_t = 0,
        n_pairs_check: Count_t = DEFAULT_N_PAIRS_CHECK,
        extra_diameter_homology_equivalence: PositiveFloat = DEFAULT_EXTRA_DIAM_EQUIVALENCE,
        filter_column_homology_equivalence: bool = True,
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
        target_loop_death = target_loop_class.death

        source_lifetime = source_loop_class.death - source_loop_class.birth
        target_lifetime = target_loop_class.death - target_loop_class.birth
        max_lifetime = max(source_lifetime, target_lifetime)

        max_column_diameter = None
        if filter_column_homology_equivalence:
            if extra_diameter_homology_equivalence < 0:
                raise ValueError("extra_diameter_homology_equivalence must be >= 0")
            max_column_diameter = (
                max(source_loop_death, target_loop_death)
                + float(extra_diameter_homology_equivalence) * max_lifetime
            )

        assert self.boundary_matrix_d1 is not None
        is_equivalent = check_homological_equivalence(
            source_loops=source_loops,
            target_loops=target_loops,
            boundary_matrix_d1=self.boundary_matrix_d1,
            n_pairs_check=n_pairs_check,
            max_column_diameter=max_column_diameter,
        )
        return (source_class_idx, target_class_idx, is_equivalent)

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
        top_k: Count_t = 1,
        noise_scale: float = DEFAULT_NOISE_SCALE,
        n_reps_per_loop: Count_t = DEFAULT_N_REPS_PER_LOOP,
        life_pct: float = DEFAULT_LIFE_PCT,
        n_cocycles_used: Count_t = DEFAULT_N_COCYCLES_USED,
        n_force_deviate: Count_t = DEFAULT_N_FORCE_DEVIATE,
        k_yen: Count_t = DEFAULT_K_YEN,
        loop_lower_t_pct: float = 2.5,
        loop_upper_t_pct: float = 97.5,
        n_pairs_check_equivalence: Count_t = DEFAULT_N_PAIRS_CHECK_EQUIVALENCE,
        extra_diameter_homology_equivalence: PositiveFloat = DEFAULT_EXTRA_DIAM_EQUIVALENCE,
        filter_column_homology_equivalence: bool = True,
        n_max_workers: Count_t = DEFAULT_N_MAX_WORKERS,
        k_neighbors_check_equivalence: Count_t = DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
        method_geometric_equivalence: LoopDistMethod = DEFAULT_LOOP_DIST_METHOD,
        verbose: bool = False,
        progress_main: Progress | None = None,
        use_log_display: bool = False,
        use_parallel: bool = False,
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

        if use_parallel:
            self._bootstrap_parallel(
                adata=adata,
                n_bootstrap=n_bootstrap,
                thresh=thresh,
                top_k=top_k,
                noise_scale=noise_scale,
                n_reps_per_loop=n_reps_per_loop,
                life_pct=life_pct,
                n_cocycles_used=n_cocycles_used,
                n_force_deviate=n_force_deviate,
                k_yen=k_yen,
                loop_lower_t_pct=loop_lower_t_pct,
                loop_upper_t_pct=loop_upper_t_pct,
                n_pairs_check_equivalence=n_pairs_check_equivalence,
                extra_diameter_homology_equivalence=extra_diameter_homology_equivalence,
                filter_column_homology_equivalence=filter_column_homology_equivalence,
                n_max_workers=n_max_workers,
                k_neighbors_check_equivalence=k_neighbors_check_equivalence,
                method_geometric_equivalence=method_geometric_equivalence,
                verbose=verbose,
                progress_main=progress_main,
                **nei_kwargs,
            )
            return

        if not use_log_display:
            console = Console()
            if progress_main is None:
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
            logger.add(
                lambda s: console.print(s, end=""),
                colorize=False,
                level="TRACE",
                format="<green>{time:YYYY/MM/DD HH:mm:ss}</green> | {level.icon} - <level>{message}</level>",
            )

        if progress_main is None:
            progress_main = Progress(disable=use_log_display)

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
                                    extra_diameter_homology_equivalence=extra_diameter_homology_equivalence,
                                    filter_column_homology_equivalence=filter_column_homology_equivalence,
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

    def _bootstrap_parallel(
        self,
        adata: AnnData,
        n_bootstrap: Count_t,
        thresh: Diameter_t | None = None,
        top_k: int = 1,
        noise_scale: float = DEFAULT_NOISE_SCALE,
        n_reps_per_loop: int = DEFAULT_N_REPS_PER_LOOP,
        life_pct: float = DEFAULT_LIFE_PCT,
        n_cocycles_used: int = DEFAULT_N_COCYCLES_USED,
        n_force_deviate: int = DEFAULT_N_FORCE_DEVIATE,
        k_yen: int = DEFAULT_K_YEN,
        loop_lower_t_pct: float = 2.5,
        loop_upper_t_pct: float = 97.5,
        n_pairs_check_equivalence: int = DEFAULT_N_PAIRS_CHECK_EQUIVALENCE,
        extra_diameter_homology_equivalence: PositiveFloat = DEFAULT_EXTRA_DIAM_EQUIVALENCE,
        filter_column_homology_equivalence: bool = True,
        n_max_workers: int = DEFAULT_N_MAX_WORKERS,
        k_neighbors_check_equivalence: int = DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
        method_geometric_equivalence: LoopDistMethod = DEFAULT_LOOP_DIST_METHOD,
        verbose: bool = False,
        progress_main: Progress | None = None,
        **nei_kwargs,
    ) -> None:
        assert self.boundary_matrix_d1 is not None

        results = run_bootstrap_pipeline(
            n_bootstrap=n_bootstrap,
            adata=adata,
            meta=self.meta,
            original_loop_classes=self.selected_loop_classes,
            original_boundary_matrix_d1=self.boundary_matrix_d1,
            n_max_workers=n_max_workers,
            verbose=verbose,
            progress=progress_main,
            thresh=thresh,
            noise_scale=noise_scale,
            top_k=top_k,
            n_reps_per_loop=n_reps_per_loop,
            life_pct=life_pct,
            n_cocycles_used=n_cocycles_used,
            n_force_deviate=n_force_deviate,
            k_yen=k_yen,
            loop_lower_t_pct=loop_lower_t_pct,
            loop_upper_t_pct=loop_upper_t_pct,
            k_neighbors_check_equivalence=k_neighbors_check_equivalence,
            method_geometric_equivalence=method_geometric_equivalence,
            n_pairs_check_equivalence=n_pairs_check_equivalence,
            extra_diameter_homology_equivalence=extra_diameter_homology_equivalence,
            filter_column_homology_equivalence=filter_column_homology_equivalence,
            **nei_kwargs,
        )

        assert self.meta.bootstrap is not None
        assert self.meta.bootstrap.indices_resample is not None
        assert self.bootstrap_data is not None

        for result in results:
            self.meta.bootstrap.indices_resample.append(result.indices_resample)
            if result.persistence_diagram is not None:
                self.bootstrap_data.persistence_diagrams.append(
                    result.persistence_diagram  # type: ignore[arg-type]
                )
            if result.cocycles is not None:
                self.bootstrap_data.cocycles.append(result.cocycles)  # type: ignore[arg-type]
            self.bootstrap_data.selected_loop_classes.append(result.loop_classes)

            for source_idx, matches in result.matches.items():
                if source_idx not in self.bootstrap_data.loop_tracks:
                    self.bootstrap_data.loop_tracks[source_idx] = LoopTrack(
                        source_class_idx=source_idx, matches=[]
                    )
                self.bootstrap_data.loop_tracks[source_idx].matches.extend(matches)

            self.bootstrap_data.num_bootstraps += 1

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
