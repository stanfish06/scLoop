# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from numba import jit
from anndata import AnnData
from ..data.types import IndexListDownSample, SizeDownSample, EmbeddingMethod
from pydantic import validate_call
from scipy.spatial.distance import directed_hausdorff
import numpy as np

@jit(nopython=True)
def _sample_impl(data: np.ndarray, seed_idx: int, n: int) -> np.ndarray:
    indices = np.zeros(n, dtype=np.int64)
    indices[0] = seed_idx
    min_dists = np.full(len(data), np.inf)
    
    for i in range(1, n):
        last_point = data[indices[i-1]]
        dists = np.sum((data - last_point) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[indices[:i]] = -1
        indices[i] = np.argmax(min_dists)
    
    return indices.tolist()

@validate_call(config={"arbitrary_types_allowed": True})
def sample(adata: AnnData, embedding_method: EmbeddingMethod, n: SizeDownSample) -> IndexListDownSample:
    """
    topology-aware downsampling
    """
    assert n <= adata.shape[0]
    assert f"X_{embedding_method}" in adata.obsm
    downsample_embedding = adata.obsm[f"X_{embedding_method}"]
    assert type(downsample_embedding) is np.ndarray
    return _sample_impl(downsample_embedding, 0, n)
