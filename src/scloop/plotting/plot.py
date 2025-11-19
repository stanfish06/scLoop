import glasbey
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call
from scipy.stats import binom, gamma

from .containers import HomologyData


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_distribution(
    data: HomologyData,
    mode: str,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5, 5)))
        created_fig = True
    else:
        created_fig = False

    if mode == "lifetime":
        lifetime_full = data.persistence_diagram
        if data.persistence_diagram_boot:
            lifetime_full = np.concatenate(
                [
                    lifetime_full,
                    np.concatenate(data.persistence_diagram_boot),
                ]
            )
        lifetime_full = lifetime_full[:, 1] - lifetime_full[:, 0]
        _, bins, _ = ax.hist(lifetime_full, bins="fd", density=True)
        params = data.parameters["gamma_fit"]
        ax.plot(
            np.linspace((bins[0] + bins[1]) / 2, np.max(lifetime_full), 1000),
            gamma.pdf(
                np.linspace((bins[0] + bins[1]) / 2, np.max(lifetime_full), 1000),
                params[0],
                loc=params[1],
                scale=params[2],
            ),
            **kwargs,
        )
        ax.axvline(
            gamma.ppf(
                0.95,
                params[0],
                loc=params[1],
                scale=params[2],
            ),
            0,
            1,
            color="red",
            **kwargs,
        )
    elif mode == "presence":
        occurance, count = np.unique(
            [len(track["loops"]) - 1 for src_loop, track in data.tracks.items()],
            return_counts=True,
        )
        mean_presence_prob = data.parameters["binom_fit"]
        ax.bar(occurance, count / np.sum(count), **kwargs)
        ax.bar(
            range(data.n_booted),
            binom.pmf(range(data.n_booted), data.n_booted, mean_presence_prob),
            **kwargs,
        )
        ax.axvline(
            binom.ppf(0.95, data.n_booted, mean_presence_prob),
            0,
            1,
            color="red",
        )

    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_tracks(
    data: HomologyData,
    mode: str,
    track_ids: list = [],
    ax: Axes | None = None,
    components: list = [0, 1],
    s: float = 1,
    **kwargs,
) -> Axes:
    n_tracks = len(track_ids)
    if n_tracks > 0:
        # cmap = glasbey.create_palette(palette_size=n_tracks)
        block_size = 5
        cmap = glasbey.create_block_palette(block_sizes=[block_size]*n_tracks)
        cmap = [cmap[i:i+block_size] for i in range(0, len(cmap), block_size)]
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5, 5)))
        created_fig = True
    else:
        created_fig = False

    if mode == "persistence_diagram":
        ax.scatter(
            data.persistence_diagram[:, 0],
            data.persistence_diagram[:, 1],
            color="lightgray",
            s=s,
            **kwargs,
        )
        for ph_boot in data.persistence_diagram_boot:
            ax.scatter(
                ph_boot[:, 0],
                ph_boot[:, 1],
                color="lightgray",
                s=s,
                **kwargs,
            )
        for i, src_tid in enumerate(track_ids):
            tracks = data.tracks[src_tid]["loops"]
            for tid in tracks:
                if tid[0] == 0:
                    ax.scatter(
                        data.persistence_diagram[tid[1], 0],
                        data.persistence_diagram[tid[1], 1],
                        color=cmap[i][int(np.floor(block_size/2))],
                        s=s,
                        **kwargs,
                    )
                else:
                    ax.scatter(
                        data.persistence_diagram_boot[tid[0] - 1][tid[1], 0],
                        data.persistence_diagram_boot[tid[0] - 1][tid[1], 1],
                        color=cmap[i][int(np.floor(block_size/2))],
                        s=s,
                        **kwargs,
                    )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        ax.set_xlim(max(min_val, 0), max_val)
        ax.set_ylim(max(min_val, 0), max_val)
        ax.plot(
            [max(min_val, 0), max_val],
            [max(min_val, 0), max_val],
            color="red",
            linestyle="--",
        )
        num_ticks = 6
        ticks = np.linspace(max(min_val, 0), max_val, num_ticks)
        tick_labs = [
            round(i, 3)
            for i in np.linspace(max(min_val, 0), max_val, num_ticks)
        ]
        ax.set_xticks(ticks, tick_labs)
        ax.set_yticks(ticks, tick_labs)
    elif mode == "lifetime_plot":
        lifetime_full = data.persistence_diagram
        if data.persistence_diagram_boot:
            lifetime_full = np.concatenate(
                [
                    lifetime_full,
                    np.concatenate(data.persistence_diagram_boot),
                ]
            )
        lifetime_full = lifetime_full[:, 1] - lifetime_full[:, 0]
        idx = np.vstack(
            [
                np.repeat(0, data.persistence_diagram.shape[0]),
                np.arange(data.persistence_diagram.shape[0]),
            ]
        ).T
        if data.persistence_diagram_boot:
            idx = np.concatenate(
                [
                    idx,
                    np.concatenate(
                        [
                            np.vstack(
                                [np.repeat(i + 1, ph.shape[0]), np.arange(ph.shape[0])]
                            ).T
                            for i, ph in enumerate(data.persistence_diagram_boot)
                        ]
                    ),
                ]
            )
        sort_idx = np.argsort(np.argsort(lifetime_full))
        ax.barh(
            sort_idx,
            lifetime_full,
            color="lightgray",
            linewidth=0,
            **kwargs,
        )
        for i, src_tid in enumerate(track_ids):
            tracks = data.tracks[src_tid]["loops"]
            loc_idx = []
            for tid in tracks:
                loc_idx.append(
                    np.where(np.logical_and(idx[:, 0] == tid[0], idx[:, 1] == tid[1]))[
                        0
                    ][0]
                )
            ax.barh(
                sort_idx[loc_idx],
                lifetime_full[loc_idx],
                color=cmap[i][int(np.floor(block_size/2))],
                linewidth=0,
                **kwargs,
            )
    elif mode == "loops_on_data":
        # TODO: allow custom color key
        ax.scatter(
            data.data_visualization[:, components[0]],
            data.data_visualization[:, components[1]],
            color="lightgray",
            s=s,
            **kwargs,
        )
        for i, src_tid in enumerate(track_ids):
            tracks = data.tracks[src_tid]["loops"]
            loops_plot = []
            for tid in tracks:
                if tid[0] == 0:
                    loops_plot.extend(data.loops_coords_visualization[tid[1]])
                else:
                    loops_plot.extend(data.loops_coords_visualization_boot[tid[0]-1][tid[1]])
            for j, loop in enumerate(loops_plot):
                ax.plot(
                    loop[:, components[0]],
                    loop[:, components[1]],
                    color=cmap[i][j % block_size],
                    **kwargs,
                )

    if created_fig:
        fig.tight_layout()

    return ax


#TODO: a thought, draw confidence bands for each track
