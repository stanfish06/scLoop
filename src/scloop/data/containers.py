# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from .metadata import ScloopMeta
from .analysis_containers import BootstrapAnalysis, HodgeAnalysis
from anndata import AnnData
import numpy as np

'''
store core homology data and associated analysis data
'''
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HomologyData:
    meta: ScloopMeta
    persistence_diagram: np.ndarray | None = None
    loop_representatives: list[list[np.ndarray]] | None = None
    boundary_matrix: tuple | None = None
    bootstrap_data: BootstrapAnalysis | None = None
    hodge_data: HodgeAnalysis | None = None

    def _compute_homology(self, adata: AnnData, thresh=None):
        pass

    def _compute_boundary_matrix(self, thresh=None):
        pass

    def _compute_loop_representatives(self):
        pass
