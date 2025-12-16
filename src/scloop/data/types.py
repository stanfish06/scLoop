# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

FeatureSelectionMethod = Literal["hvg", "delve", "none"]
EmbeddingMethod = Literal["pca", "diffmap", "scvi"]
EmbeddingNeighbors = Literal["pca", "scvi"]
LoopDistMethod = Literal["hausdorff", "frechet"]

Index_t = Annotated[int, Field(ge=0)]
Size_t = Annotated[int, Field(ge=0)]
Diameter_t = Annotated[float, Field(ge=0)]
SizeDownSample = Annotated[
    int, Field(ge=2, description="Sample to this number of cells")
]
# need at least 2 points to compute PH. Maybe also set an upper bound later as it is not feasible to compute PH on a lot of points
IndexListDownSample: TypeAlias = Annotated[
    list[Index_t],
    Field(min_length=2, description="Downsampled indices for PH computation"),
]
IndexListDistMatrix: TypeAlias = Annotated[
    list[Index_t],
    Field(
        min_length=2,
        description="Corresponding vertex index for each column of a distance matrix",
    ),
]
IndexListSimplex: TypeAlias = Annotated[
    list[Index_t],
    Field(min_length=0, description="Unique indicies for simplicies"),
]

# TODO: make a type for boundary matrix. Restrict matrix size for efficient computation
