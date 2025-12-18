# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import warnings

import numpy as np
import scanpy as sc
from anndata import AnnData
from pydantic import validate_call

from ..data.metadata import PreprocessMeta, ScloopMeta
from ..data.types import EmbeddingMethod, EmbeddingNeighbors, FeatureSelectionMethod
from .downsample import sample

__all__ = ["prepare_adata"]


@validate_call(config={"arbitrary_types_allowed": True})
def _normalize_and_select_hvg(
    adata: AnnData,
    do_norm: bool = True,
    compute_hvg: bool = True,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    batch_key: str = "sample_labels",
    subset: bool = True,
):
    """
    normalize_and_select_hvg(adata, do_norm, compute_hvg, target_sum, n_top_genes, batch_key)

    Perform preprocessing on the input anndata.

    Parameters
    ----------
    adata: anndata
        Input anndata object to be processed.
    do_norm: bool
        Whether to perform library size normalization and log1p transformation.
    compute_hvg: bool
        Whether to compute highly variable genes (HVG). (seurat_v3)
    target_sum: int
        Target sum for library size normalization.
    n_top_genes: int
        Number of top highly variable genes to select.
    batch_key: str
        Key for batch information in adata.obs.
    """
    done_hvg = "hvg" in adata.uns
    done_norm = "log1p" in adata.uns
    counts_available = "counts" in adata.layers
    if do_norm:
        if done_norm:
            warnings.warn(
                "adata has already undergone log1p transformation, nothing will be perfomed on X",
                UserWarning,
            )
        else:
            if not np.all(np.equal(np.mod(adata.X.data, 1), 0)):
                raise ValueError(
                    "adata.X contains non integer values, check if it is library normalized"
                )
            if not counts_available and not done_norm:
                print(
                    "counts layer is not available and X is not log1p transformed. Copy X to counts before normalization"
                )
                adata.layers["counts"] = adata.X.copy()
                counts_available = True
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)

    if compute_hvg:
        if done_hvg:
            warnings.warn(
                "adata has already undergone HVG selection, nothing will be perfomed on X",
                UserWarning,
            )
        else:
            if not counts_available:
                raise ValueError("counts layer is not available, cannot compute HVG")
            if batch_key not in adata.obs:
                warnings.warn(
                    f"batch_key {batch_key} not found in adata.obs. Will compute HVG on all cells.",
                    UserWarning,
                )
                batch_key = None

            if not adata.raw:
                print("no raw in adata, so save X to raw before computing hvg")
                adata.raw = adata

            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                subset=subset,
                flavor="seurat_v3",
                layer="counts",
                batch_key=batch_key,
            )


@validate_call(config={"arbitrary_types_allowed": True})
def prepare_adata(
    adata: AnnData,
    library_normalization: bool = True,
    target_sum: float = 1e4,
    feature_selection_method: FeatureSelectionMethod = "hvg",
    batch_key: str = "sample_labels",
    n_top_genes: int = 2000,
    embedding_method: EmbeddingMethod = "diffmap",
    embedding_neighbors: EmbeddingNeighbors = "pca",
    scale_before_pca=False,
    n_pca_comps: int = 100,
    n_neighbors: int = 25,
    n_diffusion_comps: int = 25,
    scvi_key: str = "X_scvi",
    downsample: bool = True,
    n_downsample: int = 1000,
    embedding_downsample: EmbeddingMethod | None = None,
    groupby_downsample: str | None = None,
    random_state_downsample: int = 0,
    copy: bool = False,
):
    """
    prepare_adata(adata, n_comps, n_neighbors, use_highly_variable, compute_diffmap, n_dcs)

    Prepare anndata for topological loop analysis.

    Parameters
    ----------
    adata: anndata
        Input anndata object, should be log-normalized with HVGs marked.
    copy: bool
        Whether to make a copy of adata.
    feature_selection_method : {'hvg', 'delve', 'none'}, default='hvg'
        Method for feature selection:
        - 'hvg': Highly variable genes
        - 'delve': DELVE feature selection
        - 'none': No feature selection
    """
    adata = adata.copy() if copy else adata

    needs_pca = "X_pca" not in adata.obsm and "pca" in (
        embedding_method,
        embedding_neighbors,
    )
    needs_diffmap = "X_diffmap" not in adata.obsm and embedding_method == "diffmap"
    needs_hvg = feature_selection_method == "hvg"
    needs_scvi = embedding_method == "scvi" or embedding_neighbors == "scvi"

    _normalize_and_select_hvg(
        adata,
        library_normalization,
        needs_hvg,
        target_sum,
        n_top_genes,
        batch_key,
        subset=False,
    )

    if needs_pca:
        if scale_before_pca:
            sc.pp.scale(adata)
        sc.pp.pca(adata, n_comps=n_pca_comps, use_highly_variable=needs_hvg)

    if needs_diffmap:
        sc.pp.neighbors(
            adata,
            method="gauss",
            n_neighbors=n_neighbors,
            key_added="neighbors_scloop",
            use_rep=embedding_method if embedding_neighbors != "pca" else None,
        )
        sc.tl.diffmap(
            adata, n_comps=n_diffusion_comps, neighbors_key="neighbors_scloop"
        )
        # first component of diffusion map represent local density
        adata.obsm["X_diffmap"] = adata.obsm["X_diffmap"][:, 1:]

    if needs_scvi:
        if scvi_key not in adata.obsm:
            raise ValueError(f"scvi key {scvi_key} does not exist in adata.obsm")

    """
    ========= downsample =========
    - minimize bottleneck distance
    - keep rare cell types
    ==============================
    """
    if embedding_downsample is None:
        embedding_downsample = embedding_method
    if downsample:
        indices_downsample = sample(
            adata=adata,
            embedding_method=embedding_downsample,
            groupby=groupby_downsample,
            n=n_downsample,
            random_state=random_state_downsample,
        )
    else:
        indices_downsample = None

    preprocess_meta = PreprocessMeta(
        library_normalized=library_normalization,
        target_sum=target_sum,
        feature_selection_method=feature_selection_method,
        batch_key=batch_key if feature_selection_method == "hvg" else None,
        n_top_genes=n_top_genes if feature_selection_method == "hvg" else None,
        embedding_method=embedding_method,
        embedding_neighbors=embedding_neighbors
        if embedding_method in ("pca", "diffmap")
        else None,
        scale_before_pca=scale_before_pca,
        n_pca_comps=n_pca_comps if needs_pca else None,
        n_neighbors=n_neighbors,
        n_diffusion_comps=n_diffusion_comps if needs_diffmap else None,
        scvi_key=scvi_key if needs_scvi else None,
        indices_downsample=indices_downsample,
        num_vertices=len(indices_downsample)
        if indices_downsample is not None
        else adata.shape[0],
    )

    if "scloop_meta" not in adata.uns:
        adata.uns["scloop_meta"] = ScloopMeta(preprocess=preprocess_meta)
    else:
        scloop_meta = adata.uns["scloop_meta"]
        if isinstance(scloop_meta, dict):
            scloop_meta = ScloopMeta(**scloop_meta)
        scloop_meta.preprocess = preprocess_meta
        adata.uns["scloop_meta"] = scloop_meta

    return adata if copy else None
