# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass

from ..data.containers import HomologyData
from ..data.metadata import CrossDatasetMatchingMeta
from ..data.types import Index_t
from .data_modules import nnRegressorDataModule
from .mlp import MLPregressor
from .nf import NeuralODEregressor


def _all_has_homology_data(adata_list: list[AnnData]):
    for i, adata in enumerate(adata_list):
        if "scloop" not in adata.uns:
            raise ValueError(f"adata {i} has no loop data")


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CrossDatasetMatcher:
    adatas: Annotated[list[AnnData], AfterValidator(_all_has_homology_data)]
    meta: CrossDatasetMatchingMeta
    homology_data_list: list[HomologyData] = Field(default_factory=list)
    mapping_model: MLPregressor | NeuralODEregressor | None = None

    def __post_init__(self):
        for adata in self.adatas:
            self.homology_data_list.append(adata.uns["scloop"])

    def _train_reference_mapping(self, **model_kwargs):
        ref_idx = self.meta.reference_idx
        ref_adata = self.adatas[ref_idx]

        X = np.array(ref_adata.obsm[self.meta.shared_embedding_key])
        Y = np.array(ref_adata.obsm[self.meta.reference_embedding_keys[ref_idx]])

        data_module = nnRegressorDataModule(x=X, y=Y)

        match self.meta.model_type:
            case "mlp":
                self.mapping_model = MLPregressor(data=data_module, **model_kwargs)
            case "nf":
                import torch

                t_span = torch.linspace(0, 1, 2)
                self.mapping_model = NeuralODEregressor(
                    data=data_module, t_span=t_span, **model_kwargs
                )
            case _:
                raise ValueError(f"unknown model_type: {self.meta.model_type}")

        max_epochs = model_kwargs.pop("max_epochs", 100)
        self.mapping_model.fit(max_epochs=max_epochs)

    def _transform_all_to_reference(self):
        ref_idx = self.meta.reference_idx
        ref_key = self.meta.reference_embedding_keys[ref_idx]

        for dataset_idx, adata in enumerate(self.adatas):
            if dataset_idx == ref_idx:
                ref_emb = adata.obsm[ref_key]
            else:
                shared_emb = np.array(adata.obsm[self.meta.shared_embedding_key])
                assert self.mapping_model is not None
                ref_emb = self.mapping_model.predict_new(shared_emb)
            adata.obsm["scloop_mapping"] = ref_emb

    def _compute_pairwise_distances(
        self,
        source_datset_idx: Index_t,
        target_dataset_idx: Index_t,
        include_bootstrap: bool = True,
        method: str = "hausdorff",
    ):
        # ideally should include all bootstrap loops? if too costly to run, try limit the number
        pass

    def _permutation_test(self, n_permutations: int = 1000):
        pass

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()
