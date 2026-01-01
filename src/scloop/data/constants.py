# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from .types import LoopDistMethod

CROSS_MATCH_KEY = "X_scloop_alilgned"
CROSS_MATCH_RESULT_KEY = "scloop_cross_match"

DEFAULT_N_MAX_WORKERS: int = 8
DEFAULT_N_HODGE_COMPONENTS: int = 10
DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING: int = 10
DEFAULT_LOOP_DIST_METHOD: LoopDistMethod = "frechet"
DEFAULT_TIMEOUT_EIGENDECOMPOSITION: float = 60.0
DEFAULT_MAXITER_EIGENDECOMPOSITION: int = 10000

NUMERIC_EPSILON: float = 1e-10

SCLOOP_UNS_KEY: str = "scloop"
SCLOOP_META_UNS_KEY: str = "scloop_meta"
SCLOOP_NEIGHBORS_KEY: str = "neighbors_scloop"
