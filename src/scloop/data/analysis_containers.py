# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from scipy.stats import false_discovery_control, fisher_exact, gamma
from scipy.stats.contingency import odds_ratio

from .base_components import LoopClass
from .types import Count_t, Index_t, MultipleTestCorrectionMethod, PositiveFloat, Size_t
from .utils import loops_to_coords


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
    # it is possible to have one-to-many matches (need a way to select best match)
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
    fisher_presence_results: (
        tuple[
            list[PositiveFloat],
            list[PositiveFloat],
            list[PositiveFloat],
            list[PositiveFloat],
        ]
        | None
    ) = None
    gamma_persistence_results: (
        tuple[list[PositiveFloat], list[PositiveFloat]] | None
    ) = None
    gamma_null_params: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None

    def _get_track_embedding(
        self, idx_track: Index_t, embedding: np.ndarray
    ) -> list[np.ndarray]:
        assert idx_track < len(self.loop_tracks)
        loops = []
        for boot_id, loop_id in self.loop_tracks[idx_track].track_ipairs:
            if boot_id < len(self.selected_loop_classes) and loop_id < len(
                self.selected_loop_classes[boot_id]
            ):
                loop_class = self.selected_loop_classes[boot_id][loop_id]
                if loop_class is not None and loop_class.representatives is not None:
                    loops.extend(
                        loops_to_coords(
                            embedding=embedding,
                            loops_vertices=loop_class.representatives,
                        )
                    )
        return loops

    def _get_loop_embedding(
        self, idx_bootstrap: Index_t, idx_loop: Index_t, embedding: np.ndarray
    ) -> list[np.ndarray]:
        if idx_bootstrap < len(self.selected_loop_classes) and idx_loop < len(
            self.selected_loop_classes[idx_bootstrap]
        ):
            loop_class = self.selected_loop_classes[idx_bootstrap][idx_loop]
            if loop_class is not None and loop_class.representatives is not None:
                return loops_to_coords(
                    embedding=embedding,
                    loops_vertices=loop_class.representatives,
                )
        return []

    def _analyze_track_loop_classes(
        self,
        idx_track: Index_t,
        source_loop_class: LoopClass,
        embedding: np.ndarray,
        values_vertices: np.ndarray,
    ):
        hodge_analysis = self.loop_tracks[idx_track].hodge_analysis
        assert hodge_analysis is not None
        assert idx_track in self.loop_tracks
        loop_class: LoopClassAnalysis = LoopClassAnalysis.from_super(
            super_obj=source_loop_class,
            embedding=embedding,
            values_vertices=values_vertices,
        )
        hodge_analysis.selected_loop_classes.append(loop_class)
        for boot_id, loop_id in self.loop_tracks[idx_track].track_ipairs:
            if boot_id < len(self.selected_loop_classes) and loop_id < len(
                self.selected_loop_classes[boot_id]
            ):
                loop_class_base = self.selected_loop_classes[boot_id][loop_id]
                assert loop_class_base is not None
                loop_class: LoopClassAnalysis = LoopClassAnalysis.from_super(
                    super_obj=loop_class_base,
                    embedding=embedding,
                    values_vertices=values_vertices,
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
    ) -> tuple[
        list[PositiveFloat],
        list[PositiveFloat],
        list[PositiveFloat],
        list[PositiveFloat],
    ]:
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

        return (
            probs_presence,
            odds_ratio_presence,
            pvalues_raw_presence,
            pvalues_corrected_presence,
        )

    def gamma_test_persistence(
        self,
        selected_loop_classes: list,
        method_pval_correction: MultipleTestCorrectionMethod,
    ) -> tuple[
        list[PositiveFloat],
        list[PositiveFloat],
        tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None,
    ]:
        if len(self.persistence_diagrams) == 0:
            return ([], [], None)

        lifetimes_bootstrap = []
        for diag in self.persistence_diagrams:
            if len(diag) <= 1:
                continue
            births = np.asarray(diag[1][0])
            deaths = np.asarray(diag[1][1])
            lifetimes_bootstrap.append(deaths - births)

        if len(lifetimes_bootstrap) == 0:
            return ([], [], None)

        lifetimes_bootstrap_arr = np.concatenate(lifetimes_bootstrap)
        lifetimes_bootstrap_arr = lifetimes_bootstrap_arr[
            np.isfinite(lifetimes_bootstrap_arr) & (lifetimes_bootstrap_arr > 0)
        ]
        if lifetimes_bootstrap_arr.size == 0:
            return ([], [], None)

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
        return (
            pvalues_raw_persistence,
            pvalues_corrected_persistence,
            self.gamma_null_params,
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoopClassAnalysis(LoopClass):
    coordinates_vertices: list[np.ndarray] | None = None
    coordinates_edges: list[np.ndarray] | None = None
    edge_gradient_raw: list[np.ndarray] | None = None
    edge_embedding_raw: list[np.ndarray] | None = None
    edge_embedding_smooth: list[np.ndarray] | None = None

    @classmethod
    def from_super(
        cls, super_obj: LoopClass, embedding: np.ndarray, values_vertices: np.ndarray
    ):
        assert super_obj.representatives is not None
        super_kwargs = super_obj.model_dump()  # type: ignore[reportAttributeAccessIssue]
        coordinates_vertices = loops_to_coords(
            embedding=embedding, loops_vertices=super_obj.representatives
        )
        coordinates_edges = [
            (emb[0:-1, :] + emb[1:, :]) / 2 for emb in coordinates_vertices
        ]
        edge_gradient_raw = loops_to_coords(
            embedding=values_vertices, loops_vertices=super_obj.representatives
        )
        edge_gradient_raw = [
            (vals[0:-1, :] + vals[1:, :]) / 2 for vals in edge_gradient_raw
        ]
        return cls(
            **super_kwargs,
            coordinates_vertices=coordinates_vertices,
            coordinates_edges=coordinates_edges,
            edge_gradient_raw=edge_gradient_raw,
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HodgeAnalysis:
    hodge_eigenvalues: list | None = None
    hodge_eigenvectors: list | None = None
    edges_masks_loop_classes: list[list[np.ndarray]] = Field(default_factory=list)
    selected_loop_classes: list[LoopClassAnalysis] = Field(default_factory=list)

    def _embed_edges(self):
        if self.hodge_eigenvectors is None:
            return

        hodge_evecs = np.array(self.hodge_eigenvectors)

        for loop_idx, loop in enumerate(self.selected_loop_classes):
            if loop.edge_gradient_raw is None:
                continue

            edge_masks = self.edges_masks_loop_classes[loop_idx]
            loop.edge_embedding_raw = []

            for rep_idx, edge_mask in enumerate(edge_masks):
                edge_gradients = loop.edge_gradient_raw[rep_idx]
                edge_evec_values = edge_mask.astype(np.float64) @ hodge_evecs.T
                edge_embedding = (
                    edge_gradients[:, :, None] * edge_evec_values[:, None, :]
                )
                loop.edge_embedding_raw.append(edge_embedding)

    def _smoothening_edge_embedding(self):
        # update edge_embedding_smooth: gaussian smooth of edge embedding using the coordinates_edges
        pass

    def _trajectory_identification(self):
        # identify trajectories using raw/smooth edge emebedding
        pass
