# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import jit
from pydantic import validate_call

from ..data.types import EmbeddingMethod, IndexListDownSample, SizeDownSample

__all__ = ["sample"]


@jit(nopython=True)
def _sample_impl(
    data: np.ndarray,
    class_labels: np.ndarray,
    classes: np.ndarray,
    seed_indices: np.ndarray,
    n: int,
) -> np.ndarray:
    # selected observation indices
    indices = np.zeros(n, dtype=np.int64)
    min_dists = np.full(len(data), np.inf)

    num_seeds = len(seed_indices)
    num_classes = len(classes)
    indices[0] = seed_indices[0]

    # for each newly added points, recompute and select out-of-bag hausdorff point
    for i in range(1, n):
        last_point = data[indices[i - 1]]
        dists = np.sum((data - last_point) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[indices[:i]] = -1
        if i >= num_seeds:
            class_indicies = np.where(class_labels == classes[i % num_classes])[0]
            next_idx = class_indicies[np.argmax(min_dists[class_indicies])]
            # if this class is exhausted, fall back to reguler sampling
            if min_dists[next_idx] == -1:
                next_idx = np.argmax(min_dists)
            indices[i] = next_idx
        else:
            indices[i] = seed_indices[i]

    return indices


@validate_call(config={"arbitrary_types_allowed": True})
def sample(
    adata: AnnData,
    groupby: str | None,
    embedding_method: EmbeddingMethod,
    n: SizeDownSample,
    random_state: int = 0,
) -> IndexListDownSample:
    """
    Topology-preserving downsampling using greedy farthest-point sampling.

    Args:
        adata: AnnData object containing the data
        groupby: column in adata.obs for class-balanced sampling, or None
        embedding_method: which embedding to use from adata.obsm
        n: number of points to sample
        random_state: random seed for reproducibility

    Returns:
        list of indices into adata.obs for the downsampled points
    """
    assert n <= adata.shape[0]
    assert f"X_{embedding_method}" in adata.obsm
    downsample_embedding = adata.obsm[f"X_{embedding_method}"]
    assert type(downsample_embedding) is np.ndarray

    if groupby is None:
        class_labels = np.zeros(adata.shape[0], dtype=np.int64)
        classes = np.array([0])
        seed_indices = np.array([np.random.randint(adata.shape[0])])
    else:
        assert type(adata.obs) is pd.DataFrame
        assert groupby in adata.obs.columns
        class_labels, classes = pd.factorize(adata.obs.loc[:, groupby])
        classes = np.arange(len(classes), dtype=np.int64)
        seed_indices = []
        np.random.seed(random_state)
        for c in classes:
            seed_indices.append(np.random.choice(np.where(class_labels == c)[0]))
        seed_indices = np.array(seed_indices)

    return _sample_impl(
        downsample_embedding, class_labels, classes, seed_indices, n
    ).tolist()
