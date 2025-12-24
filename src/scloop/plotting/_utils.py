# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call

from ..data.containers import HomologyData
from ..data.types import PositiveFloat


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def _create_figure_standard(
    figsize: tuple[PositiveFloat, PositiveFloat] = (5, 5),
    dpi: PositiveFloat = 300,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
) -> Axes:
    kwargs_axes_local = dict(kwargs_axes or {})
    rect = kwargs_axes_local.pop("rect", None)
    fig = plt.figure(figsize=figsize, dpi=dpi, **(kwargs_figure or {}))
    if rect is not None:
        ax: Axes = fig.add_axes(rect, **kwargs_axes_local)
    else:
        ax = fig.add_subplot(111, **kwargs_axes_local)
        fig.tight_layout(**(kwargs_layout or {}))
    return ax


def _get_homology_data(adata: AnnData, key_homology: str) -> HomologyData:
    assert adata.uns[key_homology] is not None
    assert type(adata.uns[key_homology]) is HomologyData
    return adata.uns[key_homology]
