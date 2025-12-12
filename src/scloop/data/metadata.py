# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from pydantic import BaseModel

from .types import (
    EmbeddingMethod,
    EmbeddingNeighbors,
    FeatureSelectionMethod,
    IndexListDownSample,
    Size_t,
)


class PreprocessMeta(BaseModel):
    library_normalized: bool
    target_sum: float
    feature_selection_method: FeatureSelectionMethod
    batch_key: str | None = None
    n_top_genes: int | None = None
    embedding_method: EmbeddingMethod | None = None
    embedding_neighbors: EmbeddingNeighbors | None = None
    scale_before_pca: bool
    n_pca_comps: int | None = None
    n_neighbors: int
    n_diffusion_comps: int | None = None
    scvi_key: str | None = None
    indices_downsample: IndexListDownSample | None = None
    num_vertices: Size_t | None = None


class ScloopMeta(BaseModel):
    preprocess: PreprocessMeta | None = None
