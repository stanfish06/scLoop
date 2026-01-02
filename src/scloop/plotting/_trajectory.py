# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from anndata import AnnData
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call

from ..data.constants import DEFAULT_DPI, DEFAULT_FIGSIZE
from ..data.types import Index_t, PositiveFloat
from ._utils import _create_figure_standard, _get_homology_data

__all__ = ["plot_trajectory"]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_trajectory(
    adata: AnnData,
    track_id: Index_t,
    basis: str,
    key_homology: str = "scloop",
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

    if hasattr(hodge, "trajectories") and hodge.trajectories:
        colors = ["tab:red", "tab:blue"]
        for i, traj_raw in enumerate(hodge.trajectories):
            color = colors[i % len(colors)]

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
                xy=p2,
                xytext=p1,
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
