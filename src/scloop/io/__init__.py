# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    from anndata import AnnData

    from ..data.containers import HomologyData

from ..data.constants import SCLOOP_UNS_KEY

SCHEMA_VERSION = "1.0.0"


def save_scloop(
    adata: AnnData,
    filepath: str | Path,
    compress: bool = True,
    overwrite: bool = False,
) -> None:
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"{filepath} exists. Use overwrite=True to replace.")

    if SCLOOP_UNS_KEY not in adata.uns:
        raise ValueError(f"No scloop data found in adata.uns['{SCLOOP_UNS_KEY}']")

    hd: HomologyData = adata.uns[SCLOOP_UNS_KEY]

    with h5py.File(filepath, "w") as f:
        schema_grp = f.create_group("_scloop_schema")
        schema_grp.attrs["version"] = SCHEMA_VERSION

        hd.to_hdf5_group(f, compress=compress)


def load_scloop(
    filepath: str | Path,
    adata: AnnData | None = None,
) -> HomologyData:
    from ..data.containers import HomologyData

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found")

    with h5py.File(filepath, "r") as f:
        if "_scloop_schema" not in f:
            raise ValueError("Not a valid scloop HDF5 file (missing schema)")

        version = f["_scloop_schema"].attrs["version"]
        if version != SCHEMA_VERSION:
            raise ValueError(f"Schema version {version} not supported")

        hd = HomologyData.from_hdf5_group(f)

    if adata is not None:
        adata.uns[SCLOOP_UNS_KEY] = hd

    return hd


__all__ = ["save_scloop", "load_scloop"]
