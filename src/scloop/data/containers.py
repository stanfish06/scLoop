# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from scipy.spatial import distance_matrix
from .metadata import ScloopMeta
from .analysis_containers import BootstrapAnalysis, HodgeAnalysis
from anndata import AnnData
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csr_matrix
from .ripser_lib import ripser
import numpy as np

"""
store core homology data and associated analysis data
"""


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HomologyData:
    meta: ScloopMeta
    persistence_diagram: list[np.ndarray] | None = None
    loop_representatives: list[list[np.ndarray]] | None = None
    sparse_pairwise_distance_matrix: csr_matrix | None = None
    boundary_matrix: tuple | None = None
    bootstrap_data: BootstrapAnalysis | None = None
    hodge_data: HodgeAnalysis | None = None

    def _compute_homology(self, adata: AnnData, thresh=None, **nei_kwargs):
        assert self.meta.preprocess is not None
        assert self.meta.preprocess.embedding_method is not None
        self.sparse_pairwise_distance_matrix = radius_neighbors_graph(
            X=adata.obsm[f"X_{self.meta.preprocess.embedding_method}"],
            radius=thresh,
            **nei_kwargs,
        )
        result = ripser(
            distance_matrix=self.sparse_pairwise_distance_matrix,
            modulus=2,
            dim_max=1,
            threshold=thresh,
            do_cocyles=True,
        )
        self.persistence_diagram = result.res.births_and_deaths_by_dim

    def _compute_boundary_matrix(self, thresh=None):
        pass

    def _compute_loop_representatives(self):
        pass
