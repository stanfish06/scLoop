# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from scipy.stats import false_discovery_control, fisher_exact
from scipy.stats.contingency import odds_ratio

from .types import Count_t, MultipleTestCorrectionMethod, PositiveFloat


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
    birth_root: float
    death_root: float
    matches: list[LoopMatch] = Field(default_factory=list)

    @property
    # it is possible to have one-to-many matches
    def n_matches(self) -> Count_t:
        return len({m.idx_bootstrap for m in self.matches})


@dataclass
class BootstrapAnalysis:
    num_bootstraps: int = 0
    persistence_diagrams: list[list] = Field(default_factory=list)
    cocycles: list[list] = Field(default_factory=list)
    loop_representatives: list[list[list[list[int]]]] = Field(default_factory=list)
    loop_tracks: dict[int, LoopTrack] = Field(default_factory=dict)

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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HodgeAnalysis:
    loop_id: str
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    loops_edges_embedding: list[np.ndarray] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)


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
