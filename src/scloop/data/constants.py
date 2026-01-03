# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from .types import LoopDistMethod, PositiveFloat

CROSS_MATCH_KEY = "X_scloop_alilgned"
CROSS_MATCH_RESULT_KEY = "scloop_cross_match"

DEFAULT_FIGSIZE: tuple[PositiveFloat, PositiveFloat] = (5, 5)
DEFAULT_DPI: PositiveFloat = 300

DEFAULT_N_MAX_WORKERS: int = 8
DEFAULT_N_HODGE_COMPONENTS: int = 10
DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING: int = 10
DEFAULT_LOOP_DIST_METHOD: LoopDistMethod = "frechet"
DEFAULT_TIMEOUT_EIGENDECOMPOSITION: float = 60.0
DEFAULT_MAXITER_EIGENDECOMPOSITION: int = 10000

NUMERIC_EPSILON: float = 1e-10

DEFAULT_LIFE_PCT: float = 0.1
DEFAULT_N_PAIRS_CHECK: int = 10
DEFAULT_N_PERMUTATIONS: int = 1000
DEFAULT_CUTOFF_PVAL: float = 0.05
DEFAULT_MAX_COLUMNS_BOUNDARY_MATRIX: int = 10000
DEFAULT_N_BOOTSTRAP: int = 10

DEFAULT_NOISE_SCALE: float = 1e-3
DEFAULT_N_REPS_PER_LOOP: int = 4
DEFAULT_N_COCYCLES_USED: int = 3
DEFAULT_N_FORCE_DEVIATE: int = 4
DEFAULT_K_YEN: int = 8
DEFAULT_N_PAIRS_CHECK_EQUIVALENCE: int = 4
# typically one neighbor is sufficient for checking
DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE: int = 1
DEFAULT_EXTRA_DIAM_EQUIVALENCE: float = 1.0

DEFAULT_WEIGHT_HODGE: float = 0.5
DEFAULT_HALF_WINDOW: int = 2

SCLOOP_UNS_KEY: str = "scloop"
SCLOOP_META_UNS_KEY: str = "scloop_meta"
SCLOOP_NEIGHBORS_KEY: str = "neighbors_scloop"

DEFAULT_BATCH_KEY: str = "sample_labels"
DEFAULT_SCVI_KEY: str = "X_scvi"

# simple look-up for jupyter output width
# set JUPYTER_COLUMNS to adjust rich console width
from IPython.display import Javascript

js_code = """
const outputJL = document.querySelector(".jp-OutputArea");
const outputJN = document.querySelector(".output_area");
const output = outputJL || outputJN;
const testStr = document.createElement("span");
testStr.textContent = "DOG";
document.body.appendChild(testStr);
const pixelWidthPerChar = testStr.offsetWidth / testStr.textContent.length;
const outputColumns = output.offsetWidth / pixelWidthPerChar;
element.append(outputColumns)
"""
CURRENT_JUPYTER_OUTPUT_WIDTH = Javascript(js_code)

import os

DEFAULT_RICH_CONSOLE_WIDTH: str = "100"
DEFAULT_RICH_CONSOLE_HEIGHT: str = "27"

if "JUPYTER_COLUMNS" not in os.environ:
    os.environ["JUPYTER_COLUMNS"] = DEFAULT_RICH_CONSOLE_WIDTH
if "JUPYTER_LINES" not in os.environ:
    os.environ["JUPYTER_LINES"] = DEFAULT_RICH_CONSOLE_HEIGHT
