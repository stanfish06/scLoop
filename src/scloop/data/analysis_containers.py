# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from pynndescent import NNDescent
from scipy.stats import false_discovery_control, fisher_exact, gamma
from scipy.stats.contingency import odds_ratio

from ..computing import compute_weighted_hodge_embedding
from .base_components import LoopClass, PersistenceTestResult, PresenceTestResult
from .constants import NUMERIC_EPSILON
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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoopMatch:
    idx_bootstrap: int
    target_class_idx: int
    geometric_distance: Optional[float] = None
    neighbor_rank: Optional[int] = None
    extra: dict = Field(default_factory=dict)


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


@dataclass
class BootstrapAnalysis:
    num_bootstraps: Size_t = 0
    persistence_diagrams: list[list] = Field(default_factory=list)
    cocycles: list[list] = Field(default_factory=list)
    selected_loop_classes: list[list[LoopClass | None]] = Field(default_factory=list)
    loop_tracks: dict[int, LoopTrack] = Field(default_factory=dict)
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
            odds_ratio_presence.append(odds_ratio(np.array(tbl)).statistic)
            res = fisher_exact(table=tbl, alternative="greater")
            pvalues_raw_presence.append(res.pvalue)  # type: ignore[attr-defined]
        match method_pval_correction:
            case "bonferroni":
                n_tests = len(pvalues_raw_presence)
                pvalues_corrected_presence = [p * n_tests for p in pvalues_raw_presence]
            case "benjamini-hochberg":
                pvalues_corrected_presence = false_discovery_control(
                    pvalues_raw_presence, method="bh"
                ).tolist()
            case _:
                raise ValueError(f"{method_pval_correction} unsupported")

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

        match method_pval_correction:
            case "bonferroni":
                n_tests = len(pvalues_raw_persistence)
                pvalues_corrected_persistence = [
                    p * n_tests for p in pvalues_raw_persistence
                ]
            case "benjamini-hochberg":
                pvalues_corrected_persistence = false_discovery_control(
                    pvalues_raw_persistence, method="bh"
                ).tolist()
            case _:
                raise ValueError(f"{method_pval_correction} unsupported")

        self.gamma_null_params = (
            float(params[0]),
            float(params[1]),
            float(params[2]),
        )
        self.gamma_persistence_results = (
            pvalues_raw_persistence,
            pvalues_corrected_persistence,
        )
        return PersistenceTestResult(
            pvalues_raw=pvalues_raw_persistence,
            pvalues_corrected=pvalues_corrected_persistence,
            gamma_null_params=self.gamma_null_params,
        )


class LoopClassAnalysis(LoopClass):
    coordinates_edges: list[np.ndarray] | None = None
    edge_gradient_raw: list[np.ndarray] | None = None
    edge_embedding_raw: list[np.ndarray] | None = None
    edge_embedding_smooth: list[np.ndarray] | None = None
    valid_edge_indices_per_rep: list[list[int]] = Field(default_factory=list)
    edge_signs_per_rep: list[np.ndarray] = Field(default_factory=list)

    @property
    def coordinates_edges_all(self):
        assert self.coordinates_edges is not None
        return np.concatenate(self.coordinates_edges)

    @property
    def edge_embedding_raw_all(self):
        assert self.edge_embedding_raw is not None
        return np.concatenate(self.edge_embedding_raw)

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
            edge_gradient_raw=edge_gradient_raw,
        )


class HodgeAnalysis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    hodge_eigenvalues: list | None = None
    hodge_eigenvectors: list | None = None
    edges_masks_loop_classes: list[list[np.ndarray]] = Field(default_factory=list)
    selected_loop_classes: list[LoopClassAnalysis] = Field(default_factory=list)

    def _embed_edges(self, weight_hodge: Percent_t, half_window: int = 2):
        if self.hodge_eigenvectors is None:
            return

        hodge_evecs = np.array(self.hodge_eigenvectors)

        # TODO: looks jittable
        for loop_idx, loop in enumerate(self.selected_loop_classes):
            if loop.edge_gradient_raw is None or loop.coordinates_edges is None:
                continue

            edge_masks = self.edges_masks_loop_classes[loop_idx]
            loop.edge_embedding_raw = []

            for rep_idx, edge_mask in enumerate(edge_masks):
                valid_indices = loop.valid_edge_indices_per_rep[rep_idx]
                if not valid_indices:
                    continue
                edge_gradients = loop.edge_gradient_raw[rep_idx][valid_indices]
                loop.coordinates_edges[rep_idx] = loop.coordinates_edges[rep_idx][
                    valid_indices
                ]
                edge_evec_values = edge_mask.astype(np.float64) @ hodge_evecs.T

                # apply sign correction for edge traversal direction
                if loop.edge_signs_per_rep and rep_idx < len(loop.edge_signs_per_rep):
                    edge_signs = loop.edge_signs_per_rep[rep_idx][:, None]
                    edge_evec_values = edge_evec_values * edge_signs

                # window smoothing along loop
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

                weighted_edge_hodge = compute_weighted_hodge_embedding(
                    edge_evecs=edge_evec_smooth,
                    eigenvalues=np.array(self.hodge_eigenvalues),
                    edge_gradients=edge_gradients_smooth,
                )
                grad_1d = edge_gradients_smooth.flatten()
                edge_embedding = weighted_edge_hodge * weight_hodge + grad_1d * (
                    1 - weight_hodge
                )
                # raw edge embedding: too noisy to use
                # edge_embedding = (
                #     edge_gradients[:, :, None] * edge_evec_values[:, None, :]
                # )
                loop.edge_embedding_raw.append(edge_embedding)

    def _smoothening_edge_embedding(self, n_neighbors: Count_t = 10):
        """weighted-knn-smoothing of edge embedding

        Parameters
        ----------
        n_neighbors : positive int
        """
        coordinates_edges_all = np.concatenate(
            [loops.coordinates_edges_all for loops in self.selected_loop_classes]
        )
        edge_embedding_raw_all = np.concatenate(
            [loops.edge_embedding_raw_all for loops in self.selected_loop_classes]
        )
        search_index = NNDescent(coordinates_edges_all)
        for loops in self.selected_loop_classes:
            loops.edge_embedding_smooth = []
            assert loops.edge_embedding_raw is not None
            assert loops.coordinates_edges is not None
            for coords in loops.coordinates_edges:
                nn_indices, nn_distances = search_index.query(
                    query_data=coords, k=n_neighbors
                )
                assert nn_indices.shape == (coords.shape[0], n_neighbors)
                assert nn_distances.shape == nn_indices.shape
                length_scale = np.median(nn_distances, axis=1, keepdims=True) + 1e-8
                nn_similarities = np.exp(-nn_distances / length_scale)
                neighbor_embeddings = edge_embedding_raw_all[nn_indices]
                nn_weights = nn_similarities / nn_similarities.sum(
                    axis=1, keepdims=True
                )
                if neighbor_embeddings.ndim == 2:
                    smoothed = (neighbor_embeddings * nn_weights).sum(axis=1)
                else:
                    smoothed = (neighbor_embeddings * nn_weights[:, :, None]).sum(
                        axis=1
                    )
                loops.edge_embedding_smooth.append(smoothed)

    def _trajectory_identification(self):
        """Identify trajectories using raw/smooth edge emebedding

        Parameters
        ----------
        param_name : type
        Description of parameter.

        Returns
        -------
        return_type
        Description of return value.
        """

        pass
