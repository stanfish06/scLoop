# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import false_discovery_control, fisher_exact, gamma
from scipy.stats.contingency import odds_ratio

from .types import Count_t, Index_t, MultipleTestCorrectionMethod, PositiveFloat, Size_t


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoopMatch:
    idx_bootstrap: int
    birth_bootstrap: float
    death_bootstrap: float
    target_class_idx: int
    geometric_distance: Optional[float] = None
    neighbor_rank: Optional[int] = None
    extra: dict = Field(default_factory=dict)


@dataclass
class LoopTrack:
    source_class_idx: int
    birth_root: PositiveFloat
    death_root: PositiveFloat
    matches: list[LoopMatch] = Field(default_factory=list)
    hodge_analysis: HodgeAnalysis | None = None

    @property
    # it is possible to have one-to-many matches
    def n_matches(self) -> Count_t:
        return len({m.idx_bootstrap for m in self.matches})

    @property
    def lifetime(self) -> PositiveFloat:
        return self.death_root - self.birth_root

    def _compute_hodge_eigendecomposition(
        self, hodge_matrix: csr_matrix, n_components: int = 10, normalized: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if hodge_matrix is None:
            logger.warning("hodge_matrix too small for eigendecomposition (shape < 2).")
            return None
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
            eigenvalues, eigenvectors = eigsh(hodge_matrix, k=k, which="SM")
            sort_idx = np.argsort(eigenvalues)
            return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
        except Exception as e:
            logger.error(f"Eigendecomposition failed: {e}")
            return None


@dataclass
class BootstrapAnalysis:
    num_bootstraps: Size_t = 0
    persistence_diagrams: list[list] = Field(default_factory=list)
    cocycles: list[list] = Field(default_factory=list)
    loop_representatives: list[list[list[list[int]]]] = Field(default_factory=list)
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
        self, method_pval_correction: MultipleTestCorrectionMethod
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
            lifetime = float(loop_track.lifetime)
            p_val = float(
                1 - gamma.cdf(lifetime, a=params[0], loc=params[1], scale=params[2])
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
class HodgeAnalysis:
    loop_id: Index_t
    hodge_eigenvalues: list | None = None
    hodge_eigenvectors: np.ndarray | None = None
    loops_edges_embedding: list | None = None
    pseudotime_analysis: PseudotimeAnalysis | None = None
    velociy_analysis: VelocityAnalysis | None = None

    def _smoothening_edge_embedding(self):
        pass

    def _trajectory_identification(self):
        pass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PseudotimeAnalysis:
    edge_pseudotime_deltas: np.ndarray | None = None
    pseudotime_source: str = ""
    parameters: dict = Field(default_factory=dict)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VelocityAnalysis:
    edge_velocity_deltas: np.ndarray | None = None
    velocity_source: str = ""
    parameters: dict = Field(default_factory=dict)
