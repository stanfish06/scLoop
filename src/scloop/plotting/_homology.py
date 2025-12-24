# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import glasbey
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call

from ..data.analysis_containers import BootstrapAnalysis
from ..data.containers import HomologyData
from ..data.types import Index_t, PositiveFloat
from ._utils import _create_figure_standard, _get_homology_data

__all__ = [
    "hist_lifetimes",
    "bar_lifetimes",
    "persistence_diagram",
    "loops",
]


# ugly function, fix that
def _get_track_loop(
    data: HomologyData, track_id: int
) -> list[tuple[int, int, float, float]]:
    if data.bootstrap_data is None:
        return []
    track = data.bootstrap_data.loop_tracks.get(track_id)
    if track is None:
        return []
    tracked_pairs = []
    if track_id < len(data.selected_loop_classes):
        loop_class = data.selected_loop_classes[track_id]
        if loop_class is not None:
            tracked_pairs.append(
                (0, track.source_class_idx, loop_class.birth, loop_class.death)
            )
    for m in track.matches:
        if m.idx_bootstrap < len(data.bootstrap_data.selected_loop_classes):
            if m.target_class_idx < len(
                data.bootstrap_data.selected_loop_classes[m.idx_bootstrap]
            ):
                boot_loop_class = data.bootstrap_data.selected_loop_classes[
                    m.idx_bootstrap
                ][m.target_class_idx]
                if boot_loop_class is not None:
                    tracked_pairs.append(
                        (
                            m.idx_bootstrap + 1,
                            m.target_class_idx,
                            boot_loop_class.birth,
                            boot_loop_class.death,
                        )
                    )
    return tracked_pairs


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def hist_lifetimes(
    adata: AnnData,
    key_homology: str = "scloop",
    dimension_homology: Index_t = 1,
    show_bootstrap: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: float = 300,
    ax: Axes | None = None,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_hist: dict | None = None,
) -> Axes:
    kwargs_axes = kwargs_axes or {}
    kwargs_axes.setdefault("rect", (0, 0, 1, 1))
    kwargs_axes.setdefault("aspect", "auto")

    data = _get_homology_data(adata=adata, key_homology=key_homology)

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
    lifetime_full = data.persistence_diagram
    assert lifetime_full is not None
    assert dimension_homology < len(lifetime_full)

    if show_bootstrap and data.bootstrap_data is not None:
        bootstrap_data: BootstrapAnalysis = data.bootstrap_data
        lifetime_bootstrap: list = bootstrap_data.persistence_diagrams
        if len(lifetime_bootstrap) > 0:
            lifetime_full = np.concatenate(
                [
                    lifetime_full[dimension_homology],
                    np.concatenate(
                        [p[dimension_homology] for p in lifetime_bootstrap], axis=1
                    ),
                ],
                axis=1,
            )
    lifetime_full = lifetime_full[1] - lifetime_full[0]
    _, bins, _ = ax.hist(lifetime_full, **(kwargs_hist or {}))

    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def bar_lifetimes(
    adata: AnnData,
    key_homology: str = "scloop",
    track_ids: list[Index_t] | None = None,
    ax: Axes | None = None,
    *,
    dimension_homology: Index_t = 1,
    show_bootstrap: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: float = 300,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    **kwargs,
) -> Axes:
    kwargs_axes = kwargs_axes or {}
    kwargs_axes.setdefault("rect", (0, 0, 1, 1))
    kwargs_axes.setdefault("aspect", "auto")

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

    lifetime_full = data.persistence_diagram
    assert lifetime_full is not None
    assert dimension_homology < len(lifetime_full)

    lifetimes = []
    indices = []
    base_diag = lifetime_full[dimension_homology]
    base_lifetimes = np.asarray(base_diag[1]) - np.asarray(base_diag[0])
    lifetimes.append(base_lifetimes)
    indices.append(
        np.vstack(
            [
                np.zeros_like(base_lifetimes, dtype=int),
                np.arange(base_lifetimes.size, dtype=int),
            ]
        ).T
    )

    if show_bootstrap and data.bootstrap_data is not None:
        bootstrap_data: BootstrapAnalysis = data.bootstrap_data
        for idx_boot, diag in enumerate(bootstrap_data.persistence_diagrams):
            if dimension_homology >= len(diag):
                continue
            boot_diag = diag[dimension_homology]
            boot_lifetimes = np.asarray(boot_diag[1]) - np.asarray(boot_diag[0])
            lifetimes.append(boot_lifetimes)
            indices.append(
                np.vstack(
                    [
                        np.full_like(boot_lifetimes, idx_boot + 1, dtype=int),
                        np.arange(boot_lifetimes.size, dtype=int),
                    ]
                ).T
            )

    if len(lifetimes) == 0:
        return ax

    lifetime_full_arr = np.concatenate(lifetimes)

    sort_idx = np.argsort(lifetime_full_arr)
    pos_idx = np.argsort(sort_idx)
    ax.barh(
        pos_idx,
        lifetime_full_arr,
        color="lightgray",
        linewidth=0,
        **kwargs,
    )

    track_ids = track_ids or []
    n_tracks = len(track_ids)
    if n_tracks > 0:
        block_size = 5
        cmap = glasbey.create_block_palette(block_sizes=[block_size] * n_tracks)
        cmap = [cmap[i : i + block_size] for i in range(0, len(cmap), block_size)]
        for i, src_tid in enumerate(track_ids):
            tracked_pairs = _get_track_loop(data, src_tid)
            lifetime_tracks = []
            loc_idx = []
            for tid in tracked_pairs:
                lifetime = tid[3] - tid[2]
                loc_idx.append(
                    np.searchsorted(a=lifetime_full_arr, v=lifetime, sorter=sort_idx)
                )
                lifetime_tracks.append(lifetime)
            if len(loc_idx) == 0:
                continue
            ax.barh(
                loc_idx,
                lifetime_tracks,
                color=cmap[i][int(np.floor(block_size / 2))],
                linewidth=0,
                **kwargs,
            )

    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def persistence_diagram(
    adata: AnnData,
    key_homology: str = "scloop",
    track_ids: list[Index_t] | None = None,
    ax: Axes | None = None,
    *,
    dimension_homology: Index_t = 1,
    show_bootstrap: bool = True,
    s: float = 1,
    figsize: tuple[float, float] = (5, 5),
    dpi: float = 300,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
    kwargs_line: dict | None = None,
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

    diag = data.persistence_diagram
    assert diag is not None
    assert dimension_homology < len(diag)

    base_diag = diag[dimension_homology]
    base_births = np.asarray(base_diag[0])
    base_deaths = np.asarray(base_diag[1])

    ax.scatter(
        base_births, base_deaths, color="lightgray", s=s, **(kwargs_scatter or {})
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])

    if show_bootstrap and data.bootstrap_data is not None:
        for diag_boot in data.bootstrap_data.persistence_diagrams:
            if dimension_homology >= len(diag_boot):
                continue
            births_boot = np.asarray(diag_boot[dimension_homology][0])
            deaths_boot = np.asarray(diag_boot[dimension_homology][1])
            ax.scatter(
                births_boot,
                deaths_boot,
                color="lightgray",
                s=s,
                **(kwargs_scatter or {}),
            )
            max_val = float(
                max(
                    max_val,
                    max(np.percentile(births_boot, 90), np.percentile(deaths_boot, 90)),
                )
            )
            min_val = float(
                min(
                    min_val,
                    min(np.percentile(births_boot, 10), np.percentile(deaths_boot, 10)),
                )
            )

    track_ids = track_ids or []
    n_tracks = len(track_ids)
    if n_tracks > 0:
        block_size = 5
        cmap = glasbey.create_block_palette(block_sizes=[block_size] * n_tracks)
        cmap = [cmap[i : i + block_size] for i in range(0, len(cmap), block_size)]
        for i, src_tid in enumerate(track_ids):
            tracked_pairs = _get_track_loop(data, src_tid)
            for tid in tracked_pairs:
                if not show_bootstrap:
                    if tid[0] > 0:
                        continue
                ax.scatter(
                    tid[2],
                    tid[3],
                    color=cmap[i][int(np.floor(block_size / 2))],
                    s=s,
                    **(kwargs_scatter or {}),
                )

    num_ticks = 6
    ax.set_xlim(float(max(min_val, 0)), max_val)
    ax.set_ylim(float(max(min_val, 0)), max_val)
    ticks = np.linspace(max(min_val, 0), max_val, num_ticks)
    tick_labs = [round(i, 3) for i in np.linspace(max(min_val, 0), max_val, num_ticks)]
    ax.set_xticks(ticks, tick_labs)
    ax.set_yticks(ticks, tick_labs)

    ax.plot(
        [float(max(min_val, 0)), max_val],
        [float(max(min_val, 0)), max_val],
        color="red",
        linestyle="--",
        **(kwargs_line or {}),
    )
    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def loops(
    adata: AnnData,
    basis: str,
    key_homology: str = "scloop",
    track_ids: list[Index_t] | None = None,
    loop_ids: list[Index_t | tuple[Index_t, Index_t]] | None = None,
    components: tuple[Index_t, Index_t] | list[Index_t] = (0, 1),
    ax: Axes | None = None,
    *,
    show_bootstrap: bool = True,
    pointsize: PositiveFloat = 1,
    figsize: tuple[PositiveFloat, PositiveFloat] = (5, 5),
    dpi: PositiveFloat = 300,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
) -> Axes:
    data = _get_homology_data(adata, key_homology)
    if len(components) != 2:
        raise ValueError("components must contain exactly two entries.")

    if basis in adata.obsm:
        emb = adata.obsm[basis]
    elif f"X_{basis}" in adata.obsm:
        emb = adata.obsm[f"X_{basis}"]
    else:
        raise KeyError(f"Embedding {basis} not found in adata.obsm.")
    assert type(emb) is np.ndarray

    if track_ids is not None and loop_ids is not None:
        raise ValueError("Only one of track_ids or loop_ids can be provided.")

    selectors = loop_ids if loop_ids is not None else track_ids or []

    n_selectors = len(selectors)
    if n_selectors > 0:
        block_size = 5
        cmap = glasbey.create_block_palette(block_sizes=[block_size] * n_selectors)
        cmap = [cmap[i : i + block_size] for i in range(0, len(cmap), block_size)]
    else:
        block_size = 0
        cmap = []
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

    if data.meta.preprocess is not None and data.meta.preprocess.indices_downsample:
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

    include_bootstrap = show_bootstrap and data.bootstrap_data is not None

    def _loops_for_selector(
        selector: Index_t | tuple[Index_t, Index_t],
    ) -> list[np.ndarray]:
        if isinstance(selector, tuple) and not include_bootstrap:
            return []
        if isinstance(selector, int):
            if selector >= len(data.selected_loop_classes):
                return []
        try:
            return data._get_loop_embedding(
                selector=selector,
                embedding_alt=emb,
                include_bootstrap=include_bootstrap,
            )
        except AssertionError:
            return []

    for i, selector in enumerate(selectors):
        loops_plot = _loops_for_selector(selector)

        for j, loop in enumerate(loops_plot):
            loop = np.array(loop)
            ax.plot(
                loop[:, components[0]],
                loop[:, components[1]],
                color=cmap[i][j % block_size],
                **(kwargs_scatter or {}),
            )

    return ax
