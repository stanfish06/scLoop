# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pydantic import ConfigDict, validate_call

from ..data.types import Index_t, PositiveFloat
from ._utils import _create_figure_standard, _get_homology_data

__all__ = [
    "loop_edge_embedding",
    "loop_edge_overlay",
]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def loop_edge_embedding(
    adata: AnnData,
    track_id: Index_t,
    key_homology: str = "scloop",
    ax: Axes | None = None,
    *,
    use_smooth: bool = False,
    color_by: Literal["position", "gradient"] = "position",
    pointsize: PositiveFloat = 5,
    figsize: tuple[PositiveFloat, PositiveFloat] = (5, 5),
    dpi: PositiveFloat = 300,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
    cmap: str = "coolwarm",
) -> Axes:
    data = _get_homology_data(adata, key_homology)

    kwargs_axes = kwargs_axes or {}
    if "aspect" not in kwargs_axes:
        kwargs_axes["aspect"] = "equal"
        kwargs_axes["rect"] = (0, 0, 1, 1)

    ax = (
        _create_figure_standard(
            figsize=figsize,
            dpi=dpi,
            kwargs_figure=kwargs_figure,
            kwargs_axes=kwargs_axes,
            kwargs_layout=kwargs_layout,
        )
        if ax is None
        else ax
    )

    track = data.bootstrap_data.loop_tracks[track_id]
    hodge = track.hodge_analysis

    all_embeddings = []
    all_colors = []

    for loop_class in hodge.selected_loop_classes:
        edge_embeddings = (
            loop_class.edge_embedding_smooth
            if use_smooth
            else loop_class.edge_embedding_raw
        )

        for rep_idx, edge_emb in enumerate(edge_embeddings):
            valid_indices = loop_class.valid_edge_indices_per_rep[rep_idx]
            n_edges = len(edge_emb)

            if color_by == "position":
                colors = np.linspace(0, 1, n_edges)
            elif color_by == "gradient":
                gradients = loop_class.edge_gradient_raw[rep_idx][valid_indices]
                colors = gradients.flatten()

            all_embeddings.append(edge_emb)
            all_colors.append(colors)

    if all_embeddings:
        all_emb = np.concatenate(all_embeddings)
        all_col = np.concatenate(all_colors)
        positions = np.arange(len(all_emb))

        ax.scatter(
            positions,
            all_emb,
            c=all_col,
            s=pointsize,
            cmap=cmap,
            **(kwargs_scatter or {}),
        )
        ax.set_xlabel("Edge index")
        ax.set_ylabel("Edge embedding")

    if len(ax.collections) > 0:
        plt.colorbar(ax.collections[-1], ax=ax)

    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def loop_edge_overlay(
    adata: AnnData,
    basis: str,
    track_id: Index_t,
    key_homology: str = "scloop",
    ax: Axes | None = None,
    *,
    components: tuple[Index_t, Index_t] = (0, 1),
    use_smooth: bool = False,
    color_by: Literal["embedding", "gradient", "position"] = "embedding",
    pointsize: PositiveFloat = 10,
    figsize: tuple[PositiveFloat, PositiveFloat] = (5, 5),
    dpi: PositiveFloat = 300,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
    kwargs_plot: dict | None = None,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Axes:
    data = _get_homology_data(adata, key_homology)

    if basis in adata.obsm:
        emb = adata.obsm[basis]
    elif f"X_{basis}" in adata.obsm:
        emb = adata.obsm[f"X_{basis}"]

    kwargs_axes = kwargs_axes or {}
    if "aspect" not in kwargs_axes:
        kwargs_axes["aspect"] = "equal"
        kwargs_axes["rect"] = (0, 0, 1, 1)

    ax = (
        _create_figure_standard(
            figsize=figsize,
            dpi=dpi,
            kwargs_figure=kwargs_figure,
            kwargs_axes=kwargs_axes,
            kwargs_layout=kwargs_layout,
        )
        if ax is None
        else ax
    )

    if data.meta.preprocess and data.meta.preprocess.indices_downsample:
        vertex_indices = data.meta.preprocess.indices_downsample
    else:
        vertex_indices = list(range(emb.shape[0]))
    emb_background = emb[vertex_indices]

    ax.scatter(
        emb_background[:, components[0]],
        emb_background[:, components[1]],
        color="lightgray",
        s=1,
        **(kwargs_scatter or {}),
    )

    track = data.bootstrap_data.loop_tracks[track_id]
    hodge = track.hodge_analysis

    all_color_values = []

    for loop_class in hodge.selected_loop_classes:
        edge_embeddings = (
            loop_class.edge_embedding_smooth
            if use_smooth
            else loop_class.edge_embedding_raw
        )

        for rep_idx, edge_coords in enumerate(loop_class.coordinates_edges):
            valid_indices = loop_class.valid_edge_indices_per_rep[rep_idx]
            edge_emb = edge_embeddings[rep_idx]

            if color_by == "embedding":
                colors = edge_emb
            elif color_by == "gradient":
                gradients = loop_class.edge_gradient_raw[rep_idx][valid_indices]
                colors = gradients.flatten()
            elif color_by == "position":
                colors = np.linspace(0, 1, len(edge_coords))

            all_color_values.extend(colors)

            points = edge_coords[:, [components[0], components[1]]]
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

            vmin_local = vmin if vmin is not None else np.min(colors)
            vmax_local = vmax if vmax is not None else np.max(colors)
            norm = Normalize(vmin=vmin_local, vmax=vmax_local)

            lc = LineCollection(segments, cmap=cmap, norm=norm, **(kwargs_plot or {}))
            lc.set_array(colors[:-1])
            lc.set_linewidth(2)
            ax.add_collection(lc)

    if len(all_color_values) > 0:
        if vmin is None:
            vmin = np.min(all_color_values)
        if vmax is None:
            vmax = np.max(all_color_values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)

    return ax
