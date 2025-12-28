# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated

import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass

from ..computing.homology import compute_loop_geometric_distance
from ..data.constants import CROSS_MATCH_KEY, DEFAULT_N_MAX_WORKERS
from ..data.containers import HomologyData
from ..data.metadata import CrossDatasetMatchingMeta
from ..data.types import Count_t, Index_t, LoopDistMethod
from .data_modules import nnRegressorDataModule
from .mlp import MLPregressor
from .nf import NeuralODEregressor


def _all_has_homology_data(adata_list: list[AnnData]):
    for i, adata in enumerate(adata_list):
        if "scloop" not in adata.uns:
            raise ValueError(f"adata {i} has no loop data")


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CrossDatasetMatcher:
    adata_list: Annotated[list[AnnData], AfterValidator(_all_has_homology_data)]
    meta: CrossDatasetMatchingMeta
    homology_data_list: list[HomologyData] = Field(default_factory=list)
    mapping_model: MLPregressor | NeuralODEregressor | None = None

    def __post_init__(self):
        for adata in self.adata_list:
            self.homology_data_list.append(adata.uns["scloop"])

    def _train_reference_mapping(self, **model_kwargs):
        ref_idx = self.meta.reference_idx
        ref_adata = self.adata_list[ref_idx]

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

        for dataset_idx, adata in enumerate(self.adata_list):
            if dataset_idx == ref_idx:
                ref_emb = adata.obsm[ref_key]
            else:
                shared_emb = np.array(adata.obsm[self.meta.shared_embedding_key])
                assert self.mapping_model is not None
                ref_emb = self.mapping_model.predict_new(shared_emb)
            adata.obsm[CROSS_MATCH_KEY] = ref_emb

    def _assess_permutation_geometric_equivalence(
        self,
        source_loops_embedding: list[list[list[float]]] | list[np.ndarray],
        target_loops_embedding: list[list[list[float]]] | list[np.ndarray],
        method: LoopDistMethod = "hausdorff",
    ) -> float:
        distances_arr = compute_loop_geometric_distance(
            source_loops_embedding, target_loops_embedding, method
        )
        mean_distance = float(np.nanmean(distances_arr))
        return mean_distance

    def _compute_pairwise_distances_permutation(
        self,
        n_permute: Count_t,
        source_dataset_idx: Index_t,
        target_dataset_idx: Index_t,
        source_loop_classes: list[Index_t],
        target_loop_classes: list[Index_t],
        include_bootstrap: bool = True,
        method: LoopDistMethod = "hausdorff",
        n_max_workers: int = DEFAULT_N_MAX_WORKERS,
        verbose: bool = True,
    ):
        n_source_loop_classes = len(source_loop_classes)
        n_target_loop_classes = len(target_loop_classes)

        source_embedding = self.adata_list[source_dataset_idx].obsm[CROSS_MATCH_KEY]
        target_embedding = self.adata_list[target_dataset_idx].obsm[CROSS_MATCH_KEY]
        assert source_embedding is not None
        assert type(source_embedding) is np.ndarray
        assert target_embedding is not None
        assert type(target_embedding) is np.ndarray

        source_hd = self.homology_data_list[source_dataset_idx]
        target_hd = self.homology_data_list[target_dataset_idx]
        embedding_source_classes = [
            source_hd._get_loop_embedding(
                selector=idx_class,
                include_bootstrap=include_bootstrap,
                embedding_alt=source_embedding,
            )
            for idx_class in source_loop_classes
        ]
        embedding_target_classes = [
            target_hd._get_loop_embedding(
                selector=idx_class,
                include_bootstrap=include_bootstrap,
                embedding_alt=target_embedding,
            )
            for idx_class in target_loop_classes
        ]

        # original distances stored at index 0
        pairwise_result_matrix = np.full(
            (n_permute + 1, n_source_loop_classes, n_target_loop_classes), np.nan
        )
        with ThreadPoolExecutor(max_workers=n_max_workers) as executor:
            tasks = {}
            for idx_permute in range(n_permute + 1):
                for i in range(n_source_loop_classes):
                    for j in range(n_target_loop_classes):
                        embedding_source = embedding_source_classes[i]
                        embedding_target = embedding_target_classes[j]
                        embedding_full = embedding_source + embedding_target

                        if not embedding_full:
                            continue

                        if idx_permute > 0:
                            n_source = len(embedding_source)
                            n_total = len(embedding_full)
                            permuted_indices = np.random.permutation(n_total)
                            embedding_source = [
                                embedding_full[k] for k in permuted_indices[:n_source]
                            ]
                            embedding_target = [
                                embedding_full[k] for k in permuted_indices[n_source:]
                            ]

                        task = executor.submit(
                            self._assess_permutation_geometric_equivalence,
                            source_loops_embedding=embedding_source,
                            target_loops_embedding=embedding_target,
                            method=method,
                        )
                        tasks[task] = (idx_permute, i, j)

            for task in as_completed(tasks):
                idx_permute, i, j = tasks[task]
                distance = task.result()
                pairwise_result_matrix[idx_permute, i, j] = distance

    def _loops_cross_match(self, n_permute: Count_t = 1000):
        self._compute_pairwise_distances_permutation()

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()
