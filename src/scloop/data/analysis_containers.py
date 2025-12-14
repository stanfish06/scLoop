# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BootstrapAnalysis:
    num_bootstraps: int = 0
    persistence_diagrams: list[list] = Field(default_factory=list)
    cocycles: list[list] = Field(default_factory=list)
    loop_representatives: list[list[list[list[int]]]] = Field(default_factory=list)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HodgeAnalysis:
    loop_id: str
    eigenvalues: np.ndarray = Field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = Field(default_factory=lambda: np.array([]))
    loops_edges_embedding: list[np.ndarray] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PseudotimeAnalysis:
    edge_pseudotime_deltas: np.ndarray = Field(default_factory=lambda: np.array([]))
    pseudotime_source: str = ""
    parameters: dict = Field(default_factory=dict)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VelocityAnalysis:
    edge_velocity_deltas: np.ndarray = Field(default_factory=lambda: np.array([]))
    velocity_source: str = ""
    parameters: dict = Field(default_factory=dict)
