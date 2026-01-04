# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call

from ..data.constants import DEFAULT_DPI, DEFAULT_FIGSIZE, SCLOOP_UNS_KEY
from ..data.types import Index_t, PositiveFloat
from ._utils import _create_figure_standard, _get_homology_data

__all__ = ["plot_trajectory", "plot_gene_trends"]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_trajectory(
    adata: AnnData,
    track_id: Index_t,
    basis: str,
    key_homology: str = SCLOOP_UNS_KEY,
    ax: Axes | None = None,
    *,
    components: tuple[Index_t, Index_t] | list[Index_t] = (0, 1),
    clip_range: tuple[float, float] = (0, 100),
    pointsize: PositiveFloat = 10,
    figsize: tuple[PositiveFloat, PositiveFloat] = DEFAULT_FIGSIZE,
    dpi: PositiveFloat = DEFAULT_DPI,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
) -> Axes:
    data = _get_homology_data(adata, key_homology)

    if basis in adata.obsm:
        emb = adata.obsm[basis]
    elif f"X_{basis}" in adata.obsm:
        emb = adata.obsm[f"X_{basis}"]
    else:
        raise ValueError(f"Embedding {basis} does not exist in adata")

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
        s=pointsize,
        **(kwargs_scatter or {}),
    )

    assert data.bootstrap_data is not None
    track = data.bootstrap_data.loop_tracks[track_id]
    assert track.hodge_analysis is not None
    hodge = track.hodge_analysis

    if hodge.trajectory_analyses:
        colors = ["tab:red", "tab:blue"]
        for i, traj_analysis in enumerate(hodge.trajectory_analyses):
            color = colors[i % len(colors)]
            traj_raw = traj_analysis.trajectory_coordinates

            start_idx = int(clip_range[0] * len(traj_raw) / 100)
            end_idx = int(clip_range[1] * len(traj_raw) / 100)
            traj = traj_raw[start_idx:end_idx]

            if len(traj) < 2:
                continue

            ax.plot(
                traj[:, components[0]],
                traj[:, components[1]],
                color=color,
                linewidth=3,
                label=f"Path {i + 1}",
            )

            p1 = traj[-2, [components[0], components[1]]]
            p2 = traj[-1, [components[0], components[1]]]

            ax.annotate(
                "",
                xy=tuple(p2),
                xytext=tuple(p1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=3,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=20,
                ),
            )
        ax.legend()

    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_gene_trends(
    adata: AnnData,
    track_id: Index_t,
    gene_names: list[str] | None = None,
    key_homology: str = SCLOOP_UNS_KEY,
    ax: Axes | None = None,
    *,
    figsize: tuple[PositiveFloat, PositiveFloat] = DEFAULT_FIGSIZE,
    dpi: PositiveFloat = DEFAULT_DPI,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_plot: dict | None = None,
    cmap: str = "tab10",
) -> Axes:
    data = _get_homology_data(adata, key_homology)

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

    assert data.bootstrap_data is not None
    track = data.bootstrap_data.loop_tracks[track_id]
    assert track.hodge_analysis is not None
    hodge = track.hodge_analysis

    if not hodge.trajectory_analyses:
        return ax

    trajectory_analyses = hodge.trajectory_analyses
    all_gene_names = set()
    for traj_analysis in trajectory_analyses:
        if traj_analysis.gene_names:
            all_gene_names.update(traj_analysis.gene_names)

    if gene_names is None:
        if trajectory_analyses[0].gene_names:
            gene_names = trajectory_analyses[0].gene_names
        else:
            gene_names = list(all_gene_names)
    else:
        gene_names = [g for g in gene_names if g in all_gene_names]

    if not gene_names:
        return ax

    cm = plt.get_cmap(cmap)
    colors = [cm(i / max(1, len(gene_names) - 1)) for i in range(len(gene_names))]

    for traj_idx, traj_analysis in enumerate(trajectory_analyses):
        if (
            traj_analysis.mean_expression is None
            or traj_analysis.values_vertices is None
        ):
            continue

        if traj_analysis.gene_names is None:
            continue

        for gene_idx, gene_name in enumerate(gene_names):
            if gene_name not in traj_analysis.gene_names:
                continue

            gene_pos = traj_analysis.gene_names.index(gene_name)
            mean_expr = traj_analysis.mean_expression[gene_pos]

            if traj_analysis.trajectory_pseudotime_range is None:
                continue

            pseudotime_range = traj_analysis.trajectory_pseudotime_range
            pseudotime = np.linspace(
                pseudotime_range[0], pseudotime_range[1], len(mean_expr)
            )

            linestyle = "-" if traj_idx == 0 else "--"
            label = (
                f"{gene_name} (path {traj_idx + 1})"
                if len(trajectory_analyses) > 1
                else gene_name
            )

            ax.plot(
                pseudotime,
                mean_expr,
                color=colors[gene_idx],
                linestyle=linestyle,
                linewidth=2,
                label=label,
                **(kwargs_plot or {}),
            )

            if (
                traj_analysis.ci_lower is not None
                and traj_analysis.ci_upper is not None
            ):
                ax.fill_between(
                    pseudotime,
                    traj_analysis.ci_lower[gene_pos],
                    traj_analysis.ci_upper[gene_pos],
                    color=colors[gene_idx],
                    alpha=0.2,
                )

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Gene Expression")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    return ax
