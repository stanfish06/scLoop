# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from pynndescent import NNDescent
from scipy.stats import fisher_exact, gamma
from scipy.stats.contingency import odds_ratio

from ..computing import compute_weighted_hodge_embedding
from ..utils.pvalues import correct_pvalues
from .base_components import LoopClass, PersistenceTestResult, PresenceTestResult
from .constants import (
    DEFAULT_HALF_WINDOW,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    DEFAULT_WEIGHT_HODGE,
    NUMERIC_EPSILON,
)
from .types import (
    Count_t,
    Index_t,
    MultipleTestCorrectionMethod,
    Percent_t,
    PositiveFloat,
    Size_t,
)
from .utils import (
    loops_to_coords,
    signed_area_2d,
    smooth_along_loop_1d,
    smooth_along_loop_2d,
)

if TYPE_CHECKING:
    import h5py


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoopMatch:
    idx_bootstrap: int
    target_class_idx: int
    geometric_distance: Optional[float] = None
    neighbor_rank: Optional[int] = None


def _serialize_loop_matches(
    matches: list[LoopMatch], group: h5py.Group, compress: bool = True
) -> None:
    if not matches:
        group.attrs["_count"] = 0
        return

    group.attrs["_count"] = len(matches)
    kw = {"compression": "gzip"} if compress else {}
    group.create_dataset(
        "idx_bootstrap",
        data=np.array([m.idx_bootstrap for m in matches], dtype=np.int64),
        **kw,
    )
    group.create_dataset(
        "target_class_idx",
        data=np.array([m.target_class_idx for m in matches], dtype=np.int64),
        **kw,
    )
    geo_dists = [
        m.geometric_distance if m.geometric_distance is not None else np.nan
        for m in matches
    ]
    group.create_dataset(
        "geometric_distance", data=np.array(geo_dists, dtype=np.float64), **kw
    )
    neighbor_ranks = [
        m.neighbor_rank if m.neighbor_rank is not None else -1 for m in matches
    ]
    group.create_dataset(
        "neighbor_rank", data=np.array(neighbor_ranks, dtype=np.int64), **kw
    )


def _deserialize_loop_matches(group: h5py.Group) -> list[LoopMatch]:
    count = int(group.attrs.get("_count", 0))
    if count == 0:
        return []

    idx_bootstraps = np.asarray(group["idx_bootstrap"])
    target_class_idxs = np.asarray(group["target_class_idx"])
    geo_dists = np.asarray(group["geometric_distance"])
    neighbor_ranks = np.asarray(group["neighbor_rank"])

    matches = []
    for i in range(count):
        geo_dist = float(geo_dists[i]) if not np.isnan(geo_dists[i]) else None
        rank = int(neighbor_ranks[i]) if neighbor_ranks[i] >= 0 else None
        matches.append(
            LoopMatch(
                idx_bootstrap=int(idx_bootstraps[i]),
                target_class_idx=int(target_class_idxs[i]),
                geometric_distance=geo_dist,
                neighbor_rank=rank,
            )
        )
    return matches


@dataclass
class LoopTrack:
    source_class_idx: int
    matches: list[LoopMatch] = Field(default_factory=list)
    hodge_analysis: HodgeAnalysis | None = None

    @property
    # it is possible to have one-to-many matches (TODO: need a way to select best match)
    def n_matches(self) -> Count_t:
        return len({m.idx_bootstrap for m in self.matches})

    @property
    def track_ipairs(self) -> list[tuple[Index_t, Index_t]]:
        if self.matches is None:
            return []
        return [(m.idx_bootstrap, m.target_class_idx) for m in self.matches]

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "LoopTrack"
        group.attrs["source_class_idx"] = self.source_class_idx

        matches_grp = group.create_group("matches")
        _serialize_loop_matches(self.matches, matches_grp, compress=compress)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> LoopTrack:
        source_class_idx = int(group.attrs["source_class_idx"])  # type: ignore[arg-type]
        matches_grp: h5py.Group = group["matches"]  # type: ignore[assignment]
        matches = _deserialize_loop_matches(matches_grp)
        return cls(
            source_class_idx=source_class_idx, matches=matches, hodge_analysis=None
        )


@dataclass
class BootstrapAnalysis:
    num_bootstraps: Size_t = 0
    persistence_diagrams: list[list] = Field(default_factory=list)
    cocycles: list[list] = Field(default_factory=list)
    selected_loop_classes: list[list[LoopClass | None]] = Field(default_factory=list)
    loop_tracks: dict[Index_t, LoopTrack] = Field(default_factory=dict)
    fisher_presence_results: PresenceTestResult | None = None
    gamma_persistence_results: PersistenceTestResult | None = None

    def _get_track_embedding(
        self, idx_track: Index_t, embedding_alt: np.ndarray | None = None
    ) -> list[np.ndarray]:
        assert idx_track < len(self.loop_tracks)
        loops = []
        for boot_id, loop_id in self.loop_tracks[idx_track].track_ipairs:
            if boot_id < len(self.selected_loop_classes) and loop_id < len(
                self.selected_loop_classes[boot_id]
            ):
                loop_class = self.selected_loop_classes[boot_id][loop_id]
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
        return loops

    def _get_loop_embedding(
        self,
        idx_bootstrap: Index_t,
        idx_loop: Index_t,
        embedding_alt: np.ndarray | None = None,
    ) -> list[list[list[float]]]:
        if idx_bootstrap < len(self.selected_loop_classes) and idx_loop < len(
            self.selected_loop_classes[idx_bootstrap]
        ):
            loop_class = self.selected_loop_classes[idx_bootstrap][idx_loop]
            if loop_class is not None:
                if embedding_alt is None:
                    if loop_class.coordinates_vertices_representatives is not None:
                        return loop_class.coordinates_vertices_representatives
                else:
                    if loop_class.representatives is not None:
                        return loops_to_coords(
                            embedding=embedding_alt,
                            loops_vertices=loop_class.representatives,
                        )
        return []

    def _analyze_track_loop_classes(
        self,
        idx_track: Index_t,
        source_loop_class: LoopClass,
        values_vertices: np.ndarray,
    ):
        hodge_analysis = self.loop_tracks[idx_track].hodge_analysis
        assert hodge_analysis is not None
        assert idx_track in self.loop_tracks

        loop_class: LoopClassAnalysis = LoopClassAnalysis.from_super(
            super_obj=source_loop_class,
            values_vertices=values_vertices,
        )
        hodge_analysis.selected_loop_classes.append(loop_class)

        assert loop_class.coordinates_vertices_representatives is not None
        ref_coords = np.array(loop_class.coordinates_vertices_representatives[0])
        ref_area = signed_area_2d(ref_coords)

        for boot_id, loop_id in self.loop_tracks[idx_track].track_ipairs:
            if boot_id < len(self.selected_loop_classes) and loop_id < len(
                self.selected_loop_classes[boot_id]
            ):
                loop_class_base = self.selected_loop_classes[boot_id][loop_id]
                assert loop_class_base is not None
                loop_class: LoopClassAnalysis = LoopClassAnalysis.from_super(
                    super_obj=loop_class_base,
                    values_vertices=values_vertices,
                    ref_area=ref_area,
                )
                hodge_analysis.selected_loop_classes.append(loop_class)

    @property
    def _n_total_matches(self) -> Count_t:
        return sum([tk.n_matches for tk in self.loop_tracks.values()])

    def _contingency_table_track_to_rest(
        self, tid: int
    ) -> tuple[tuple[Count_t, Count_t], tuple[Count_t, Count_t]]:
        assert tid in self.loop_tracks
        n_matches_track = self.loop_tracks[tid].n_matches
        n_total_matches = self._n_total_matches
        return (
            (n_matches_track, n_total_matches - n_matches_track),
            (
                self.num_bootstraps - n_matches_track,
                self.num_bootstraps * (len(self.loop_tracks) - 1)
                - (n_total_matches - n_matches_track),
            ),
        )

    def fisher_test_presence(
        self, method_pval_correction: MultipleTestCorrectionMethod
    ) -> PresenceTestResult:
        assert self.num_bootstraps > 0
        probs_presence = []
        odds_ratio_presence = []
        pvalues_raw_presence = []
        for tid in self.loop_tracks.keys():
            tbl = self._contingency_table_track_to_rest(tid)
            probs_presence.append(
                float(tbl[0][0]) / (float(tbl[0][0]) + float(tbl[1][0]))
            )
            or_val = odds_ratio(np.array(tbl)).statistic
            odds_ratio_presence.append(or_val if np.isfinite(or_val) else 0.0)
            res = fisher_exact(table=tbl, alternative="greater")
            pvalues_raw_presence.append(res.pvalue)  # type: ignore[attr-defined]
        pvalues_corrected_presence = correct_pvalues(
            pvalues_raw_presence, method=method_pval_correction
        )

        return PresenceTestResult(
            probabilities=probs_presence,
            odds_ratios=odds_ratio_presence,
            pvalues_raw=pvalues_raw_presence,
            pvalues_corrected=pvalues_corrected_presence,
        )

    def gamma_test_persistence(
        self,
        selected_loop_classes: list,
        method_pval_correction: MultipleTestCorrectionMethod,
    ) -> PersistenceTestResult:
        if len(self.persistence_diagrams) == 0:
            return PersistenceTestResult(
                pvalues_raw=[], pvalues_corrected=[], gamma_null_params=None
            )

        lifetimes_bootstrap = []
        for diag in self.persistence_diagrams:
            if len(diag) <= 1:
                continue
            births = np.asarray(diag[1][0])
            deaths = np.asarray(diag[1][1])
            lifetimes_bootstrap.append(deaths - births)

        if len(lifetimes_bootstrap) == 0:
            return PersistenceTestResult(
                pvalues_raw=[], pvalues_corrected=[], gamma_null_params=None
            )

        lifetimes_bootstrap_arr = np.concatenate(lifetimes_bootstrap)
        lifetimes_bootstrap_arr = lifetimes_bootstrap_arr[
            np.isfinite(lifetimes_bootstrap_arr) & (lifetimes_bootstrap_arr > 0)
        ]
        if lifetimes_bootstrap_arr.size == 0:
            return PersistenceTestResult(
                pvalues_raw=[], pvalues_corrected=[], gamma_null_params=None
            )

        params = gamma.fit(lifetimes_bootstrap_arr, floc=0)

        pvalues_raw_persistence: list[PositiveFloat] = []
        for loop_track in self.loop_tracks.values():
            source_idx = loop_track.source_class_idx
            if source_idx < len(selected_loop_classes):
                loop_class = selected_loop_classes[source_idx]
                if loop_class is not None:
                    lifetime = float(loop_class.lifetime)
                    p_val = float(
                        1
                        - gamma.cdf(
                            lifetime, a=params[0], loc=params[1], scale=params[2]
                        )
                    )
                    pvalues_raw_persistence.append(p_val)

        pvalues_corrected_persistence = correct_pvalues(
            pvalues_raw_persistence, method=method_pval_correction
        )

        self.gamma_null_params = (
            float(params[0]),
            float(params[1]),
            float(params[2]),
        )
        self.gamma_persistence_results = PersistenceTestResult(
            pvalues_raw=pvalues_raw_persistence,
            pvalues_corrected=pvalues_corrected_persistence,
            gamma_null_params=self.gamma_null_params,
        )
        return self.gamma_persistence_results

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "BootstrapAnalysis"
        group.attrs["num_bootstraps"] = self.num_bootstraps

        slc_grp = group.create_group("selected_loop_classes")
        slc_grp.attrs["_count"] = len(self.selected_loop_classes)
        for boot_idx, loop_classes in enumerate(self.selected_loop_classes):
            boot_grp = slc_grp.create_group(str(boot_idx))
            boot_grp.attrs["_count"] = len(loop_classes)
            for lc_idx, lc in enumerate(loop_classes):
                lc_grp = boot_grp.create_group(str(lc_idx))
                if lc is None:
                    lc_grp.attrs["_is_none"] = True
                else:
                    lc_grp.attrs["_is_none"] = False
                    lc.to_hdf5_group(lc_grp, compress=compress)

        # loop_tracks: dict[int, LoopTrack]
        tracks_grp = group.create_group("loop_tracks")
        for track_id, track in self.loop_tracks.items():
            track_grp = tracks_grp.create_group(str(track_id))
            track.to_hdf5_group(track_grp, compress=compress)

        # test results
        if self.fisher_presence_results is not None:
            fisher_grp = group.create_group("fisher_presence_results")
            self.fisher_presence_results.to_hdf5_group(fisher_grp, compress=compress)

        if self.gamma_persistence_results is not None:
            gamma_grp = group.create_group("gamma_persistence_results")
            self.gamma_persistence_results.to_hdf5_group(gamma_grp, compress=compress)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> BootstrapAnalysis:
        num_bootstraps = int(group.attrs["num_bootstraps"])  # type: ignore[arg-type]

        # selected_loop_classes
        selected_loop_classes: list[list[LoopClass | None]] = []
        slc_grp: h5py.Group = group["selected_loop_classes"]  # type: ignore[assignment]
        n_boots = int(slc_grp.attrs["_count"])  # type: ignore[arg-type]
        for boot_idx in range(n_boots):
            boot_grp: h5py.Group = slc_grp[str(boot_idx)]  # type: ignore[assignment]
            n_lcs = int(boot_grp.attrs["_count"])  # type: ignore[arg-type]
            loop_classes: list[LoopClass | None] = []
            for lc_idx in range(n_lcs):
                lc_grp: h5py.Group = boot_grp[str(lc_idx)]  # type: ignore[assignment]
                if lc_grp.attrs.get("_is_none", False):
                    loop_classes.append(None)
                else:
                    loop_classes.append(LoopClass.from_hdf5_group(lc_grp))
            selected_loop_classes.append(loop_classes)

        # loop_tracks
        loop_tracks: dict[int, LoopTrack] = {}
        tracks_grp: h5py.Group = group["loop_tracks"]  # type: ignore[assignment]
        for track_id_str in tracks_grp.keys():
            track_grp: h5py.Group = tracks_grp[track_id_str]  # type: ignore[assignment]
            loop_tracks[int(track_id_str)] = LoopTrack.from_hdf5_group(track_grp)

        # test results
        fisher_presence_results = None
        if "fisher_presence_results" in group:
            fisher_grp: h5py.Group = group["fisher_presence_results"]  # type: ignore[assignment]
            fisher_presence_results = PresenceTestResult.from_hdf5_group(fisher_grp)

        gamma_persistence_results = None
        if "gamma_persistence_results" in group:
            gamma_grp: h5py.Group = group["gamma_persistence_results"]  # type: ignore[assignment]
            gamma_persistence_results = PersistenceTestResult.from_hdf5_group(gamma_grp)

        return cls(
            num_bootstraps=num_bootstraps,
            persistence_diagrams=[],
            cocycles=[],
            selected_loop_classes=selected_loop_classes,
            loop_tracks=loop_tracks,
            fisher_presence_results=fisher_presence_results,
            gamma_persistence_results=gamma_persistence_results,
        )


class LoopClassAnalysis(LoopClass):
    coordinates_edges: list[np.ndarray] | None = None
    edge_values_raw: list[np.ndarray] | None = None
    edge_gradient_raw: list[np.ndarray] | None = None
    edge_embedding_raw: list[np.ndarray] | None = None
    edge_embedding_smooth: list[np.ndarray] | None = None
    edge_involvement_raw: list[np.ndarray] | None = None
    edge_involvement_smooth: list[np.ndarray] | None = None
    valid_edge_indices_per_rep: list[list[int]] = Field(default_factory=list)
    edge_signs_per_rep: list[np.ndarray] = Field(default_factory=list)

    def _concat_property(
        self, attr_name: str, apply_filter: bool = False
    ) -> np.ndarray:
        attr = getattr(self, attr_name)
        if attr is None:
            return np.array([])

        if apply_filter and self.valid_edge_indices_per_rep:
            filtered_parts = []
            for part, indices in zip(attr, self.valid_edge_indices_per_rep):
                if len(indices) > 0:
                    filtered_parts.append(part[indices])
            if not filtered_parts:
                return (
                    np.array([]).reshape(0, *attr[0].shape[1:])
                    if attr
                    else np.array([])
                )
            return np.concatenate(filtered_parts)

        return np.concatenate(attr)

    @property
    def coordinates_edges_all(self):
        return self._concat_property("coordinates_edges", apply_filter=True)

    @property
    def edge_values_raw_all(self):
        return self._concat_property("edge_values_raw", apply_filter=True)

    @property
    def edge_gradient_raw_all(self):
        return self._concat_property("edge_gradient_raw", apply_filter=True)

    @property
    def edge_embedding_raw_all(self):
        return self._concat_property("edge_embedding_raw", apply_filter=False)

    @property
    def edge_embedding_smooth_all(self):
        return self._concat_property("edge_embedding_smooth", apply_filter=False)

    @property
    def edge_involvement_raw_all(self):
        return self._concat_property("edge_involvement_raw", apply_filter=False)

    @property
    def edge_involvement_smooth_all(self):
        return self._concat_property("edge_involvement_smooth", apply_filter=False)

    @classmethod
    def from_super(
        cls,
        super_obj: LoopClass,
        values_vertices: np.ndarray,
        ref_area: float | None = None,
    ):
        assert super_obj.representatives is not None
        assert super_obj.coordinates_vertices_representatives is not None

        coordinates_vertices = [
            np.array(coords)
            for coords in super_obj.coordinates_vertices_representatives
        ]
        representatives = [list(rep) for rep in super_obj.representatives]

        if len(coordinates_vertices) > 0:
            if ref_area is None:
                ref_area = signed_area_2d(coordinates_vertices[0])
            if abs(ref_area) > NUMERIC_EPSILON:
                for i in range(len(coordinates_vertices)):
                    if ref_area * signed_area_2d(coordinates_vertices[i]) < 0:
                        coordinates_vertices[i] = coordinates_vertices[i][::-1]
                        representatives[i] = representatives[i][::-1]

        coordinates_edges = [
            (emb[0:-1, :] + emb[1:, :]) / 2 for emb in coordinates_vertices
        ]

        if values_vertices.ndim == 1:
            values_vertices = values_vertices.reshape(-1, 1)

        edge_values_raw = loops_to_coords(
            embedding=values_vertices, loops_vertices=representatives
        )
        edge_values_raw = [
            (np.array(vals)[0:-1, :] + np.array(vals)[1:, :]) / 2
            for vals in edge_values_raw
        ]

        edge_gradient_raw = loops_to_coords(
            embedding=values_vertices, loops_vertices=representatives
        )
        edge_gradient_raw = [
            np.diff(np.array(vals), axis=0) for vals in edge_gradient_raw
        ]

        return cls(
            rank=super_obj.rank,
            birth=super_obj.birth,
            death=super_obj.death,
            cocycles=super_obj.cocycles,
            representatives=representatives,
            coordinates_vertices_representatives=[
                c.tolist() for c in coordinates_vertices
            ],
            coordinates_edges=coordinates_edges,
            edge_values_raw=edge_values_raw,
            edge_gradient_raw=edge_gradient_raw,
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TrajectoryAnalysis:
    trajectory_coordinates: np.ndarray
    trajectory_pseudotime_range: tuple[float, float] | None = None
    n_bins: int = 20

    weights_vertices: np.ndarray | None = None
    indices_vertices: np.ndarray | None = None
    values_vertices: np.ndarray | None = None
    bandwidth_vertices: float | None = None
    distances_vertices: np.ndarray | None = None

    gene_names: list[str] | None = None
    mean_expression: np.ndarray | None = None
    se_expression: np.ndarray | None = None
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None

    gam_n_splines: int = 10


class HodgeAnalysis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    hodge_eigenvalues: list | None = None
    hodge_eigenvectors: list | None = None
    edges_masks_loop_classes: list[list[np.ndarray]] = Field(default_factory=list)
    selected_loop_classes: list[LoopClassAnalysis] = Field(default_factory=list)
    trajectory_analyses: list[TrajectoryAnalysis] = Field(default_factory=list)

    def _embed_edges(
        self,
        weight_hodge: Percent_t = DEFAULT_WEIGHT_HODGE,
        half_window: int = DEFAULT_HALF_WINDOW,
    ):
        if self.hodge_eigenvectors is None:
            return

        hodge_evecs = np.array(self.hodge_eigenvectors)

        for loop_idx, loop in enumerate(self.selected_loop_classes):
            if loop.edge_gradient_raw is None or loop.coordinates_edges is None:
                continue

            edge_masks = self.edges_masks_loop_classes[loop_idx]
            loop.edge_embedding_raw = []
            loop.edge_involvement_raw = []

            for rep_idx, edge_mask in enumerate(edge_masks):
                valid_indices = loop.valid_edge_indices_per_rep[rep_idx]
                if not valid_indices:
                    continue

                edge_gradients = loop.edge_gradient_raw[rep_idx][valid_indices]
                edge_evec_values = edge_mask.astype(np.float64) @ hodge_evecs.T

                if loop.edge_signs_per_rep and rep_idx < len(loop.edge_signs_per_rep):
                    edge_signs = loop.edge_signs_per_rep[rep_idx][:, None]
                    edge_evec_values = edge_evec_values * edge_signs

                if half_window > 0:
                    edge_gradients_smooth = smooth_along_loop_1d(
                        edge_gradients.flatten().astype(np.float64), half_window
                    ).reshape(-1, 1)
                    edge_evec_smooth = smooth_along_loop_2d(
                        edge_evec_values.astype(np.float64), half_window
                    )
                else:
                    edge_gradients_smooth = edge_gradients
                    edge_evec_smooth = edge_evec_values

                weighted_edge_hodge, involvement_hodge = (
                    compute_weighted_hodge_embedding(
                        edge_evecs=edge_evec_smooth,
                        eigenvalues=np.array(self.hodge_eigenvalues),
                        edge_gradients=edge_gradients_smooth,
                    )
                )
                grad_1d = edge_gradients_smooth.flatten()
                edge_embedding = weighted_edge_hodge * weight_hodge + grad_1d * (
                    1 - weight_hodge
                )
                loop.edge_embedding_raw.append(edge_embedding)
                loop.edge_involvement_raw.append(involvement_hodge)

    def _smoothening_edge_embedding(
        self, n_neighbors: Count_t = DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING
    ):
        coordinates_edges_all = np.concatenate(
            [loop.coordinates_edges_all for loop in self.selected_loop_classes], axis=0
        )
        edge_embedding_raw_all = np.concatenate(
            [loop.edge_embedding_raw_all for loop in self.selected_loop_classes], axis=0
        )
        edge_involvement_raw_all = np.concatenate(
            [loop.edge_involvement_raw_all for loop in self.selected_loop_classes],
            axis=0,
        )

        search_index = NNDescent(coordinates_edges_all)
        for loops in self.selected_loop_classes:
            loops.edge_embedding_smooth = []
            loops.edge_involvement_smooth = []
            assert loops.edge_embedding_raw is not None
            assert loops.edge_involvement_raw is not None
            assert loops.coordinates_edges is not None

            for rep_idx, coords_raw in enumerate(loops.coordinates_edges):
                valid_idx = loops.valid_edge_indices_per_rep[rep_idx]
                if not valid_idx:
                    continue

                coords = coords_raw[valid_idx]
                nn_indices, nn_distances = search_index.query(
                    query_data=coords, k=n_neighbors
                )
                length_scale = np.median(nn_distances, axis=1, keepdims=True) + 1e-8
                nn_similarities = np.exp(-nn_distances / length_scale)
                nn_weights = nn_similarities / nn_similarities.sum(
                    axis=1, keepdims=True
                )

                neighbor_embeddings = edge_embedding_raw_all[nn_indices]
                if neighbor_embeddings.ndim == 2:
                    smoothed_emb = (neighbor_embeddings * nn_weights).sum(axis=1)
                else:
                    smoothed_emb = (neighbor_embeddings * nn_weights[:, :, None]).sum(
                        axis=1
                    )
                loops.edge_embedding_smooth.append(smoothed_emb)

                neighbor_involvements = edge_involvement_raw_all[nn_indices]
                if neighbor_involvements.ndim == 2:
                    smoothed_inv = (neighbor_involvements * nn_weights).sum(axis=1)
                else:
                    smoothed_inv = (neighbor_involvements * nn_weights[:, :, None]).sum(
                        axis=1
                    )
                loops.edge_involvement_smooth.append(smoothed_inv)

    def _trajectory_identification(
        self,
        coordinates_vertices: np.ndarray,
        values_vertices: np.ndarray,
        use_smooth: bool = True,
        percentile_threshold_involvement: Percent_t = 0,
        n_bins: int = 20,
        min_n_bins: int = 4,
        s: float = 0.1,
        padding_pct: float = 0.2,
        split_threshold: float = 0.0,
    ):
        from scipy.interpolate import splev, splprep

        coords_all = np.concatenate(
            [lc.coordinates_edges_all for lc in self.selected_loop_classes]
        )
        vals_all = np.concatenate(
            [lc.edge_values_raw_all for lc in self.selected_loop_classes]
        ).flatten()

        emb_all = np.concatenate(
            [
                lc.edge_embedding_smooth_all
                if use_smooth
                else lc.edge_embedding_raw_all
                for lc in self.selected_loop_classes
            ]
        )
        inv_all = np.concatenate(
            [
                lc.edge_involvement_smooth_all
                if use_smooth
                else lc.edge_involvement_raw_all
                for lc in self.selected_loop_classes
            ]
        )

        mask_edge_involved = inv_all > np.percentile(
            a=inv_all, q=percentile_threshold_involvement * 100
        )
        vals_involved = vals_all[mask_edge_involved]

        if len(vals_involved) == 0:
            self.trajectory_analyses = []
            return

        t_loop_start = float(np.percentile(vals_involved, 5))
        t_loop_end = float(np.percentile(vals_involved, 95))
        loop_span = t_loop_end - t_loop_start

        t_global_min = float(np.min(values_vertices))
        t_global_max = float(np.max(values_vertices))

        t_entry_min = max(t_global_min, t_loop_start - (loop_span * padding_pct))
        t_exit_max = min(t_global_max, t_loop_end + (loop_span * padding_pct))

        n_bins_stem = max(2, int(n_bins * padding_pct))

        def get_stem_points(t_start, t_end, n_bins_s):
            if t_end <= t_start:
                return [], [], []

            bins = np.linspace(t_start, t_end, n_bins_s + 1)
            stem_centers = []
            stem_weights = []
            stem_t = []

            for i in range(n_bins_s):
                m_bin = (values_vertices >= bins[i]) & (values_vertices < bins[i + 1])
                if np.any(m_bin):
                    center = np.mean(coordinates_vertices[m_bin], axis=0)
                    weight = np.sum(m_bin)
                    stem_centers.append(center)
                    stem_weights.append(weight)
                    stem_t.append((bins[i] + bins[i + 1]) / 2)

            return stem_centers, stem_weights, stem_t

        entry_centers, entry_weights, entry_t = get_stem_points(
            t_entry_min, t_loop_start, n_bins_stem
        )
        exit_centers, exit_weights, exit_t = get_stem_points(
            t_loop_end, t_exit_max, n_bins_stem
        )

        trajs = []
        traj_analyses = []
        for sign in [1, -1]:
            if sign == 1:
                mask_sign = emb_all > split_threshold
            else:
                mask_sign = emb_all < -split_threshold

            mask_arm = mask_edge_involved & mask_sign
            if not np.any(mask_arm):
                continue

            c_arm = coords_all[mask_arm]
            v_arm = vals_all[mask_arm]
            w_arm = inv_all[mask_arm]

            bins = np.linspace(v_arm.min(), v_arm.max(), n_bins + 1)
            arm_centers = []
            arm_weights = []
            arm_t = []

            for i in range(n_bins):
                m_bin = (v_arm >= bins[i]) & (v_arm < bins[i + 1])
                if np.any(m_bin):
                    weights_bin = w_arm[m_bin]
                    total_weight = np.sum(weights_bin) + NUMERIC_EPSILON
                    center = np.average(c_arm[m_bin], axis=0, weights=weights_bin)

                    arm_centers.append(center)
                    arm_weights.append(total_weight)
                    arm_t.append((bins[i] + bins[i + 1]) / 2)

            full_centers = entry_centers + arm_centers + exit_centers
            full_weights = entry_weights + arm_weights + exit_weights
            full_t = entry_t + arm_t + exit_t

            if len(full_centers) < min_n_bins:
                continue

            pts = np.array(full_centers).T
            w_pts = np.array(full_weights)

            idx_sort = np.argsort(full_t)
            pts = pts[:, idx_sort]
            w_pts = w_pts[idx_sort]
            w_pts = w_pts / np.max(w_pts)

            try:
                # TODO: how to make it more robust?
                # tck, u = splprep(pts, w=w_pts, s=s)
                tck, u = splprep(pts)
                u_fine = np.linspace(0, 1, n_bins * 10)
                traj_fine = np.array(splev(u_fine, tck)).T
                trajs.append(traj_fine)

                from scipy.spatial.distance import cdist

                edge_distances = cdist(c_arm, traj_fine).min(axis=1)
                bandwidth = float(np.percentile(edge_distances, 75))

                traj_min_t = full_t[idx_sort[0]]
                traj_max_t = full_t[idx_sort[-1]]

                traj_analysis = TrajectoryAnalysis(
                    trajectory_coordinates=traj_fine,
                    trajectory_pseudotime_range=(
                        float(traj_min_t),
                        float(traj_max_t),
                    ),
                    n_bins=n_bins,
                    bandwidth_vertices=bandwidth,
                )
                traj_analyses.append(traj_analysis)
            except Exception:
                continue

        self.trajectory_analyses = traj_analyses

    def _compute_gene_trends(
        self,
        coordinates_vertices: np.ndarray,
        gene_expression_matrix: np.ndarray,
        gene_names: list[str],
        values_vertices: np.ndarray,
        confidence_level: float = 0.95,
        bandwidth_scale: float = 1.0,
        verbose: bool = False,
    ):
        from ..analyzing.gene_trend import compute_gene_trends_for_trajectories

        if len(self.trajectory_analyses) == 0:
            if verbose:
                from loguru import logger

                logger.warning("No trajectories available for gene trend analysis")
            return

        compute_gene_trends_for_trajectories(
            trajectory_analyses=self.trajectory_analyses,
            coordinates_vertices=coordinates_vertices,
            gene_expression_matrix=gene_expression_matrix,
            gene_names=gene_names,
            values_vertices=values_vertices,
            confidence_level=confidence_level,
            bandwidth_scale=bandwidth_scale,
            verbose=verbose,
        )
