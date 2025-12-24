# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)

from pydantic import BaseModel, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from .types import Diameter_t, Index_t, PositiveFloat


class LoopClass(BaseModel):
    rank: Index_t
    birth: Diameter_t = 0.0
    death: Diameter_t = 0.0
    cocycles: list | None = None
    representatives: list[list[Index_t]] | None = None
    coordinates_vertices_representatives: list[list[list[float]]] | None = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_birth_death(self) -> Self:
        if self.birth > self.death:
            raise ValueError("loop dies before its birth")
        return self

    @property
    def lifetime(self):
        return self.death - self.birth


@dataclass
class PresenceTestResult:
    probabilities: list[PositiveFloat]
    odds_ratios: list[PositiveFloat]
    pvalues_raw: list[PositiveFloat]
    pvalues_corrected: list[PositiveFloat]


@dataclass
class PersistenceTestResult:
    pvalues_raw: list[PositiveFloat]
    pvalues_corrected: list[PositiveFloat]
    gamma_null_params: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None
