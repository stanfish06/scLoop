# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from typing import Literal, Annotated, TypeAlias
from pydantic import Field

FeatureSelectionMethod = Literal["hvg", "delve", "none"]
EmbeddingMethod = Literal["pca", "diffmap", "scvi"]
EmbeddingNeighbors = Literal["pca", "scvi"]

Index_t = Annotated[int, Field(ge=0)]
# need at least 2 points to compute PH. Maybe also set an upper bound later as it is not feasible to compute PH on a lot of points
IndexListDownSample: TypeAlias = Annotated[
    list[Index_t],
    Field(min_length=2, description="Downsampled indices for PH computation"),
]
SizeDownSample = Annotated[
    int, Field(ge=2, description="Sample to this number of cells")
]
# TODO: make a type for boundary matrix. Restrict matrix size for efficient computation
