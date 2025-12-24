# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import glasbey
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from pydantic import ConfigDict, validate_call

from ..data.analysis_containers import HodgeAnalysis, LoopClassAnalysis
from ..data.types import Index_t
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
    *,
    components: tuple[Index_t, Index_t] = (0, 1),
    use_smooth: bool = False,
    color_by: str = "loop",
    show_loop_classes: bool = True,
    figsize: tuple[float, float] = (6, 6),
    dpi: float = 300,
    pointsize: float = 5.0,
    ax: Axes | None = None,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
    cmap: str = "viridis",
) -> Axes:
    kwargs_axes = kwargs_axes or {}
    kwargs_axes.setdefault("rect", (0, 0, 1, 1))
    kwargs_axes.setdefault("aspect", "equal")

    data = _get_homology_data(adata, key_homology)

    assert data.bootstrap_data is not None, "No bootstrap data found"
    assert track_id in data.bootstrap_data.loop_tracks, f"Track {track_id} not found"

    track = data.bootstrap_data.loop_tracks[track_id]
    assert track.hodge_analysis is not None, f"No Hodge analysis for track {track_id}"

    hodge: HodgeAnalysis = track.hodge_analysis
    assert hodge.hodge_eigenvectors is not None, "No Hodge eigenvectors found"

    n_components = len(hodge.hodge_eigenvectors)
    assert components[0] < n_components and components[1] < n_components

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

    loop_classes_to_plot = (
        hodge.selected_loop_classes
        if show_loop_classes
        else (
            [hodge.selected_loop_classes[0]]
            if len(hodge.selected_loop_classes) > 0
            else []
        )
    )

    assert len(loop_classes_to_plot) > 0, "No loop classes found"

    n_loops = len(loop_classes_to_plot)
    loop_colors = (
        glasbey.create_palette(palette_size=n_loops)
        if color_by == "loop" and n_loops > 1
        else None
    )

    for loop_idx, loop_class in enumerate(loop_classes_to_plot):
        loop_class: LoopClassAnalysis

        edge_embeddings = (
            loop_class.edge_embedding_smooth
            if use_smooth
            else loop_class.edge_embedding_raw
        )
        assert edge_embeddings is not None, f"No embeddings for loop {loop_idx}"

        for rep_idx, edge_emb in enumerate(edge_embeddings):
            edge_hodge_coords = np.sum(edge_emb, axis=1)

            x_coords = edge_hodge_coords[:, components[0]]
            y_coords = edge_hodge_coords[:, components[1]]

            if color_by == "loop":
                colors = loop_colors[loop_idx] if loop_colors is not None else "C0"
            elif color_by == "position":
                colors = np.linspace(0, 1, len(x_coords))
            elif color_by == "gradient":
                assert loop_class.edge_gradient_raw is not None
                gradients = loop_class.edge_gradient_raw[rep_idx]
                colors = np.linalg.norm(gradients, axis=1)
            else:
                raise ValueError(f"Unknown color_by: {color_by}")

            scatter_kwargs = dict(kwargs_scatter or {})
            if color_by in ["position", "gradient"]:
                scatter_kwargs.setdefault("cmap", cmap)
                ax.scatter(x_coords, y_coords, c=colors, s=pointsize, **scatter_kwargs)
            else:
                ax.scatter(
                    x_coords, y_coords, color=colors, s=pointsize, **scatter_kwargs
                )

    ax.set_xlabel(f"Hodge component {components[0]}")
    ax.set_ylabel(f"Hodge component {components[1]}")
    ax.set_title(
        f"Loop {track_id} edge embedding ({'smooth' if use_smooth else 'raw'})"
    )

    if color_by in ["position", "gradient"] and len(ax.collections) > 0:
        plt.colorbar(ax.collections[-1], ax=ax, label=color_by)

    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def loop_edge_overlay(
    adata: AnnData,
    track_id: Index_t,
    key_homology: str = "scloop",
    *,
    basis: str = "diffmap",
    components: tuple[Index_t, Index_t] = (0, 1),
    hodge_component: Index_t = 0,
    use_smooth: bool = False,
    color_by: str = "hodge",
    show_background: bool = True,
    show_loop_classes: bool = True,
    figsize: tuple[float, float] = (6, 6),
    dpi: float = 300,
    pointsize: float = 10.0,
    pointsize_bg: float = 1.0,
    linewidth: float = 2.0,
    ax: Axes | None = None,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
    kwargs_plot: dict | None = None,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Axes:
    kwargs_axes = kwargs_axes or {}
    kwargs_axes.setdefault("rect", (0, 0, 1, 1))
    kwargs_axes.setdefault("aspect", "equal")

    data = _get_homology_data(adata, key_homology)

    assert data.bootstrap_data is not None, "No bootstrap data found"
    assert track_id in data.bootstrap_data.loop_tracks, f"Track {track_id} not found"

    track = data.bootstrap_data.loop_tracks[track_id]
    assert track.hodge_analysis is not None, f"No Hodge analysis for track {track_id}"

    hodge: HodgeAnalysis = track.hodge_analysis

    emb_key = f"X_{basis}"
    assert emb_key in adata.obsm, f"Embedding {emb_key} not found"
    emb = np.array(adata.obsm[emb_key])
    assert components[0] < emb.shape[1] and components[1] < emb.shape[1]

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

    if show_background:
        if data.meta.preprocess is not None and data.meta.preprocess.indices_downsample:
            vertex_indices = data.meta.preprocess.indices_downsample
        else:
            vertex_indices = list(range(emb.shape[0]))
        emb_background = emb[vertex_indices]

        ax.scatter(
            emb_background[:, components[0]],
            emb_background[:, components[1]],
            color="lightgray",
            s=pointsize_bg,
            alpha=0.5,
            **(kwargs_scatter or {}),
        )

    loop_classes_to_plot = (
        hodge.selected_loop_classes
        if show_loop_classes
        else (
            [hodge.selected_loop_classes[0]]
            if len(hodge.selected_loop_classes) > 0
            else []
        )
    )

    assert len(loop_classes_to_plot) > 0, "No loop classes found"

    if color_by == "hodge":
        assert hodge.hodge_eigenvectors is not None
        n_hodge_components = len(hodge.hodge_eigenvectors)
        assert hodge_component < n_hodge_components

    all_color_values = []

    for loop_idx, loop_class in enumerate(loop_classes_to_plot):
        loop_class: LoopClassAnalysis

        if loop_class.coordinates_edges is None:
            continue

        if color_by == "hodge":
            edge_embeddings = (
                loop_class.edge_embedding_smooth
                if use_smooth
                else loop_class.edge_embedding_raw
            )
            assert edge_embeddings is not None

        for rep_idx, edge_coords in enumerate(loop_class.coordinates_edges):
            x_coords = edge_coords[:, components[0]]
            y_coords = edge_coords[:, components[1]]

            if color_by == "hodge":
                edge_emb = edge_embeddings[rep_idx]
                hodge_values = np.sum(edge_emb[:, :, hodge_component], axis=1)
                colors = hodge_values
            elif color_by == "gradient":
                assert loop_class.edge_gradient_raw is not None
                gradients = loop_class.edge_gradient_raw[rep_idx]
                colors = np.linalg.norm(gradients, axis=1)
            elif color_by == "position":
                colors = np.linspace(0, 1, len(x_coords))
            else:
                raise ValueError(f"Unknown color_by: {color_by}")

            all_color_values.extend(colors)

            vmin_local = vmin if vmin is not None else np.min(colors)
            vmax_local = vmax if vmax is not None else np.max(colors)
            norm = Normalize(vmin=vmin_local, vmax=vmax_local)

            ax.scatter(
                x_coords,
                y_coords,
                c=colors,
                cmap=cmap,
                s=pointsize,
                norm=norm,
                zorder=10,
                **(kwargs_scatter or {}),
            )

            ax.plot(
                x_coords,
                y_coords,
                color="black",
                linewidth=linewidth,
                alpha=0.3,
                zorder=5,
                **(kwargs_plot or {}),
            )

    ax.set_xlabel(f"{basis} {components[0]}")
    ax.set_ylabel(f"{basis} {components[1]}")

    title_parts = [f"Loop {track_id}"]
    if color_by == "hodge":
        title_parts.append(f"Hodge component {hodge_component}")
    if use_smooth:
        title_parts.append("(smooth)")
    ax.set_title(" - ".join(title_parts))

    if len(all_color_values) > 0:
        if vmin is None:
            vmin = np.min(all_color_values)
        if vmax is None:
            vmax = np.max(all_color_values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_by)

    return ax
