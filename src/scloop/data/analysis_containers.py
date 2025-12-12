# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BootstrapAnalysis:
    loops_eidx_boot: list[list[list[np.ndarray]]] = Field(default_factory=list)
    persistence_diagram_boot: list[np.ndarray] = Field(default_factory=list)
    matching_df: list[pd.DataFrame] = Field(default_factory=list)
    n_booted: int = 0
    loop_rank: pd.DataFrame = Field(default_factory=pd.DataFrame)
    parameters: dict = Field(default_factory=dict)


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
