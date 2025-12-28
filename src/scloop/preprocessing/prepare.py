# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import numpy as np
import scanpy as sc
from anndata import AnnData
from loguru import logger
from pydantic import validate_call
from rich.console import Console

from ..data.constants import SCLOOP_META_UNS_KEY, SCLOOP_NEIGHBORS_KEY
from ..data.metadata import PreprocessMeta, ScloopMeta
from ..data.types import EmbeddingMethod, EmbeddingNeighbors, FeatureSelectionMethod
from .downsample import sample

__all__ = ["prepare_adata"]

console = Console()


@validate_call(config={"arbitrary_types_allowed": True})
def _normalize_and_select_hvg(
    adata: AnnData,
    do_norm: bool = True,
    compute_hvg: bool = True,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    batch_key: str | None = "sample_labels",
    subset: bool = True,
    verbose: bool = True,
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
    batch_key: str | None
        Key for batch information in adata.obs.
    """
    done_hvg = "hvg" in adata.uns
    done_norm = "log1p" in adata.uns
    counts_available = "counts" in adata.layers
    if do_norm:
        if done_norm:
            if verbose:
                logger.info("adata already normalized (log1p), skipping normalization")
        else:
            X = adata.X
            values = X.data if hasattr(X, "data") else X
            if not np.all(np.equal(np.mod(np.asarray(values), 1), 0)):
                raise ValueError(
                    "adata.X contains non integer values, check if it is library normalized"
                )
            if not counts_available and not done_norm:
                if verbose:
                    logger.info("Copying X to counts layer before normalization")
                adata.layers["counts"] = adata.X.copy()
                counts_available = True
            if verbose:
                logger.info(
                    f"Normalizing to target sum {target_sum} and applying log1p transformation"
                )
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)

    if compute_hvg:
        if done_hvg:
            if verbose:
                logger.info("adata already has HVG selection, skipping HVG computation")
        else:
            if not counts_available:
                raise ValueError("counts layer is not available, cannot compute HVG")
            if batch_key not in adata.obs:
                if verbose:
                    logger.warning(
                        f"batch_key '{batch_key}' not found in adata.obs, computing HVG on all cells"
                    )
                batch_key = None

            if not adata.raw:
                if verbose:
                    logger.info("Saving X to raw before computing HVG")
                adata.raw = adata

            if verbose:
                logger.info(
                    f"Computing top {n_top_genes} highly variable genes (seurat_v3)"
                )
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
    batch_key: str | None = "sample_labels",
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
    verbose: bool = True,
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

    if verbose:
        logger.remove()
        logger.add(
            lambda s: console.print(s, end=""),
            colorize=False,
            level="TRACE",
            format="<green>{time:YYYY/MM/DD HH:mm:ss}</green> | {level.icon} - <level>{message}</level>",
        )
        logger.info(
            f"Preparing AnnData with {adata.n_obs} cells and {adata.n_vars} genes"
        )

    needs_pca = "X_pca" not in adata.obsm and "pca" in (
        embedding_method,
        embedding_neighbors,
    )
    needs_diffmap = "X_diffmap" not in adata.obsm and embedding_method == "diffmap"
    needs_hvg = feature_selection_method == "hvg"
    needs_scvi = embedding_method == "scvi" or embedding_neighbors == "scvi"

    if verbose and (library_normalization or needs_hvg):
        logger.info("Step 1/4: Normalization and feature selection")
    _normalize_and_select_hvg(
        adata,
        library_normalization,
        needs_hvg,
        target_sum,
        n_top_genes,
        batch_key,
        subset=False,
        verbose=verbose,
    )

    if needs_pca:
        if verbose:
            logger.info("Step 2/4: Computing PCA")
        if scale_before_pca:
            if verbose:
                logger.info("Scaling data before PCA")
            sc.pp.scale(adata)
        if verbose:
            logger.info(f"Computing PCA with {n_pca_comps} components")
        sc.pp.pca(adata, n_comps=n_pca_comps, use_highly_variable=needs_hvg)
    elif "X_pca" in adata.obsm:
        if verbose:
            logger.info("Step 2/4: PCA already computed, skipping")
    else:
        if verbose:
            logger.info("Step 2/4: PCA not needed, skipping")

    if needs_diffmap:
        if verbose:
            logger.info("Step 3/4: Computing diffusion map")
            logger.info(f"Computing neighbors with n_neighbors={n_neighbors}")
        sc.pp.neighbors(
            adata,
            method="gauss",
            n_neighbors=n_neighbors,
            key_added=SCLOOP_NEIGHBORS_KEY,
            use_rep=embedding_method if embedding_neighbors != "pca" else None,
        )
        if verbose:
            logger.info(f"Computing diffusion map with {n_diffusion_comps} components")
        sc.tl.diffmap(
            adata, n_comps=n_diffusion_comps, neighbors_key=SCLOOP_NEIGHBORS_KEY
        )
        # first component of diffusion map represent local density
        adata.obsm["X_diffmap"] = adata.obsm["X_diffmap"][:, 1:]
    elif "X_diffmap" in adata.obsm:
        if verbose:
            logger.info("Step 3/4: Diffusion map already computed, skipping")
    else:
        if verbose:
            logger.info("Step 3/4: Diffusion map not needed, skipping")

    if needs_scvi:
        if scvi_key not in adata.obsm:
            raise ValueError(f"scvi key {scvi_key} does not exist in adata.obsm")
        if verbose:
            logger.info(f"Using scVI embedding from {scvi_key}")

    """
    ========= downsample =========
    - minimize bottleneck distance
    - keep rare cell types
    ==============================
    """
    if verbose:
        logger.info("Step 4/4: Downsampling")
    if embedding_downsample is None:
        embedding_downsample = embedding_method
    if downsample:
        if verbose:
            logger.info(
                f"Downsampling to {n_downsample} cells using {embedding_downsample} embedding"
            )
            if groupby_downsample:
                logger.info(f"Stratified by {groupby_downsample}")
        indices_downsample = sample(
            adata=adata,
            embedding_method=embedding_downsample,
            groupby=groupby_downsample,
            n=n_downsample,
            random_state=random_state_downsample,
        )
        if verbose:
            logger.info(f"Selected {len(indices_downsample)} cells for analysis")
    else:
        if verbose:
            logger.info("Downsampling disabled, using all cells")
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
        num_vertices=adata.shape[0],
    )

    if SCLOOP_META_UNS_KEY not in adata.uns:
        adata.uns[SCLOOP_META_UNS_KEY] = ScloopMeta(preprocess=preprocess_meta)
    else:
        scloop_meta = adata.uns[SCLOOP_META_UNS_KEY]
        if isinstance(scloop_meta, dict):
            scloop_meta = ScloopMeta(**scloop_meta)
        scloop_meta.preprocess = preprocess_meta
        adata.uns[SCLOOP_META_UNS_KEY] = scloop_meta

    if verbose:
        logger.success("AnnData preparation complete")

    return adata if copy else None
