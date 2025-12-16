# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoopMatch:
    idx_bootstrap: int
    birth_bootstrap: float
    death_bootstrap: float
    target_class_idx: int
    geometric_distance: Optional[float] = None
    neighbor_rank: Optional[int] = None
    extra: dict = Field(default_factory=dict)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoopTrack:
    source_class_idx: int
    birth_root: float
    death_root: float
    matches: list[LoopMatch] = Field(default_factory=list)

    def presence_prob(self, n_bootstraps: int) -> float:
        if n_bootstraps == 0:
            return 0.0
        hit_boots = {m.idx_bootstrap for m in self.matches}
        return len(hit_boots) / n_bootstraps


@dataclass
class BootstrapAnalysis:
    num_bootstraps: int = 0
    persistence_diagrams: list[list] = Field(default_factory=list)
    cocycles: list[list] = Field(default_factory=list)
    loop_representatives: list[list[list[list[int]]]] = Field(default_factory=list)
    loop_tracks: dict[int, LoopTrack] = Field(default_factory=dict)


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
