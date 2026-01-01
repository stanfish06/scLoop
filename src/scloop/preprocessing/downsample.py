# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import jit
from pydantic import validate_call
from pynndescent import NNDescent

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
    percent_removal_density: float = 0.025,
    n_neighbors_density: int = 50,
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

    # TODO: try mellon for better density estimation
    if percent_removal_density > 0:
        index = NNDescent(downsample_embedding, n_neighbors=n_neighbors_density)
        _, distances = index.query(downsample_embedding, k=n_neighbors_density)
        distances = distances / np.median(distances)
        d_sq = distances**2
        bandwidth_inv = d_sq.mean(axis=1, keepdims=True)
        similarities = np.exp(-d_sq * bandwidth_inv / 2)
        density = similarities.sum(axis=1)

        n_total = len(density)
        min_removal = percent_removal_density
        current_percentile = percent_removal_density / 2
        threshold = np.percentile(density, current_percentile * 100)
        while current_percentile < 1.0:
            threshold = np.percentile(density, current_percentile * 100)
            n_kept = np.sum(density >= threshold)
            if n_kept <= n_total * (1 - min_removal):
                break
            current_percentile *= 2

        valid_mask = density >= threshold
        local_indices = np.where(valid_mask)[0]
        downsample_embedding_local = downsample_embedding[local_indices]
    else:
        local_indices = np.arange(adata.shape[0])
        downsample_embedding_local = downsample_embedding

    if groupby is None:
        class_labels = np.zeros(len(local_indices), dtype=np.int64)
        classes = np.array([0])
        np.random.seed(random_state)
        seed_indices = np.array([np.random.randint(len(local_indices))])
    else:
        assert type(adata.obs) is pd.DataFrame
        assert groupby in adata.obs.columns
        class_labels_global, classes = pd.factorize(adata.obs.loc[:, groupby])
        class_labels = class_labels_global[local_indices].astype(np.int64)
        classes_present = np.unique(class_labels)
        seed_indices = []
        np.random.seed(random_state)
        for c in classes_present:
            seed_indices.append(np.random.choice(np.where(class_labels == c)[0]))
        seed_indices = np.array(seed_indices)
        classes = classes_present

    local_result = _sample_impl(
        downsample_embedding_local, class_labels, classes, seed_indices, n
    )
    return local_indices[local_result].tolist()
