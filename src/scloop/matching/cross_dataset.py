# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.stats import ttest_ind

from ..computing.homology import compute_loop_geometric_distance
from ..data.constants import CROSS_MATCH_KEY, DEFAULT_N_MAX_WORKERS
from ..data.containers import HomologyData
from ..data.metadata import CrossDatasetMatchingMeta
from ..data.types import (
    Count_t,
    Index_t,
    LoopDistMethod,
    MultipleTestCorrectionMethod,
    Percent_t,
    PositiveFloat,
    Size_t,
)
from ..utils.pvalues import correct_pvalues
from .data_modules import nnRegressorDataModule
from .mlp import MLPregressor
from .nf import NeuralODEregressor


def _all_has_homology_data(adata_list: list[AnnData]):
    for i, adata in enumerate(adata_list):
        if "scloop" not in adata.uns:
            raise ValueError(f"adata {i} has no loop data")


@dataclass
class CrossLoopMatch:
    source_dataset_idx: Index_t
    target_dataset_idx: Index_t
    source_class_idx: Index_t
    target_class_idx: Index_t
    source_class_match_embedding: list[list[list[float]]]
    target_class_match_embedding: list[list[list[float]]]
    geometric_distance: PositiveFloat
    null_distribution_geometric_distance: list[PositiveFloat]
    pvalue_permutation: Percent_t
    pvalue_corrected_permutation: Percent_t
    t_stats_match: float | None = None
    pvalue_match: Percent_t | None = None
    pvalue_corrected_match: Percent_t | None = None


@dataclass
class LoopClass:
    rank: Size_t = 0

    def __post_init__(self):
        self.parent = self

    @property
    def is_representative(self):
        return self.parent == self

    def _get_representative(self) -> LoopClass:
        # path compression
        if self.parent != self:
            self.parent = self._get_representative()
        return self.parent  # type: ignore


class LoopClassIdx(
    NamedTuple
):  # By itself, fields can be any type. Within pydantic classes, field types will be checked.
    idx_dataset: Index_t
    idx_loop_class: Index_t


@dataclass
class LoopTrack:
    loop_classes: dict[LoopClassIdx, LoopClass] = Field(default_factory=dict)

    def _union(self, idx_source: LoopClassIdx, idx_target: LoopClassIdx):
        rep_a = self.loop_classes[idx_source]._get_representative()
        rep_b = self.loop_classes[idx_target]._get_representative()
        if rep_a.rank > rep_b.rank:
            rep_b.parent = rep_a
        elif rep_a.rank == rep_b.rank:
            rep_b.parent = rep_a
            rep_a.rank += 1
        else:
            rep_a.parent = rep_b

    def _get_tracks(self) -> list[list[LoopClassIdx]]:
        tracks = {}
        for idx_loop_class, loop_class in self.loop_classes.items():
            rep = loop_class._get_representative()
            if rep not in tracks:
                tracks[rep] = []
            tracks[rep].append(idx_loop_class)
        return list(tracks.values())


@dataclass
class CrossLoopMatchResult:
    n_permute: Count_t
    matches: dict[frozenset[Index_t], list[CrossLoopMatch]] = Field(
        default_factory=dict
    )
    tracks: list[list[LoopClassIdx]] | None = None

    def _compute_tracks(self):
        """Compute track from loop matches

        Parameters
        ----------
        param_name : type
        Description of parameter.

        Returns
        -------
        return_type
        Description of return value.
        """
        tracks = LoopTrack()
        for match_list in self.matches.values():
            for loop_match in match_list:
                idx_source = LoopClassIdx(
                    idx_dataset=loop_match.source_dataset_idx,
                    idx_loop_class=loop_match.source_class_idx,
                )
                idx_target = LoopClassIdx(
                    idx_dataset=loop_match.target_dataset_idx,
                    idx_loop_class=loop_match.target_class_idx,
                )
                if idx_source not in tracks.loop_classes:
                    tracks.loop_classes[idx_source] = LoopClass()
                if idx_target not in tracks.loop_classes:
                    tracks.loop_classes[idx_target] = LoopClass()
                tracks._union(idx_source=idx_source, idx_target=idx_target)
        self.tracks = tracks._get_tracks()

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CrossDatasetMatcher:
    adata_list: Annotated[list[AnnData], AfterValidator(_all_has_homology_data)]
    meta: CrossDatasetMatchingMeta
    homology_data_list: list[HomologyData] = Field(default_factory=list)
    mapping_model: MLPregressor | NeuralODEregressor | None = None
    loop_matching_result: CrossLoopMatchResult | None = None

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

    def _get_loop_class_embedding(
        self,
        dataset_idx: Index_t,
        class_idx: Index_t,
        include_bootstrap: bool = True,
    ) -> list[list[list[float]]]:
        embedding = self.adata_list[dataset_idx].obsm[CROSS_MATCH_KEY]
        assert embedding is not None
        assert type(embedding) is np.ndarray
        hd = self.homology_data_list[dataset_idx]
        return hd._get_loop_embedding(
            selector=class_idx,
            include_bootstrap=include_bootstrap,
            embedding_alt=embedding,
        )

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
        progress: Progress | None = None,
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

        n_total_tasks = (n_permute + 1) * n_source_loop_classes * n_target_loop_classes
        task_id = None
        if progress is not None:
            task_id = progress.add_task("[cyan]Permutation tests", total=n_total_tasks)

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
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)

        return pairwise_result_matrix

    def _loops_cross_match(
        self,
        n_permute: Count_t = 1000,
        source_dataset_idx: Index_t = 0,
        target_dataset_idx: Index_t = 1,
        method: LoopDistMethod = "hausdorff",
        include_bootstrap: bool = True,
        cutoff_pval: Percent_t = 0.05,
        method_pval_correction: MultipleTestCorrectionMethod
        | None = "benjamini-hochberg",
        verbose: bool = True,
    ):
        source_hd = self.homology_data_list[source_dataset_idx]
        target_hd = self.homology_data_list[target_dataset_idx]

        source_loop_classes = list(range(len(source_hd.selected_loop_classes)))
        target_loop_classes = list(range(len(target_hd.selected_loop_classes)))

        console = Console()
        progress_main = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        logger.remove()
        logger.add(
            lambda s: console.print(s, end=""),
            colorize=False,
            level="TRACE",
            format="<green>{time:YYYY/MM/DD HH:mm:ss}</green> | {level.icon} - <level>{message}</level>",
        )

        if verbose:
            logger.info(
                f"Computing pairwise distances: {len(source_loop_classes)} source classes "
                f"x {len(target_loop_classes)} target classes, {n_permute} permutations"
            )

        with progress_main:
            pairwise_result_matrix = self._compute_pairwise_distances_permutation(
                n_permute=n_permute,
                source_dataset_idx=source_dataset_idx,
                target_dataset_idx=target_dataset_idx,
                source_loop_classes=source_loop_classes,
                target_loop_classes=target_loop_classes,
                include_bootstrap=include_bootstrap,
                method=method,
                progress=progress_main,
            )

        pairwise_result_pvalues = (
            np.sum(
                pairwise_result_matrix[1:, ...] >= pairwise_result_matrix[0, ...],
                axis=0,
            )
        ) / n_permute

        pairwise_result_pvalues_corrected = correct_pvalues(
            pairwise_result_pvalues, method=method_pval_correction
        )

        matched_mask = pairwise_result_pvalues_corrected > cutoff_pval
        unmatched_mask = ~matched_mask

        null_distributions = pairwise_result_matrix[1:, :, :]
        unmatched_nulls = null_distributions[:, unmatched_mask]
        pooled_unmatched_nulls = unmatched_nulls.ravel()
        pooled_unmatched_nulls = pooled_unmatched_nulls[
            ~np.isnan(pooled_unmatched_nulls)
        ]

        matched_indices = np.argwhere(matched_mask)
        matched_pvalues = []
        matched_t_stats = []
        for i, j in matched_indices:
            matched_null_dist = null_distributions[:, i, j]
            matched_null_dist = matched_null_dist[~np.isnan(matched_null_dist)]
            t_stats, p_val = ttest_ind(
                matched_null_dist,
                pooled_unmatched_nulls,
                equal_var=False,
                alternative="less",
            )
            matched_pvalues.append(p_val)
            matched_t_stats.append(t_stats)
        matched_pvalues_corrected = correct_pvalues(
            matched_pvalues, method=method_pval_correction
        )

        if self.loop_matching_result is None:
            self.loop_matching_result = CrossLoopMatchResult(n_permute=n_permute)

        dataset_key = frozenset([source_dataset_idx, target_dataset_idx])
        for idx, (i, j) in enumerate(matched_indices):
            loop_match = CrossLoopMatch(
                source_dataset_idx=source_dataset_idx,
                target_dataset_idx=target_dataset_idx,
                source_class_idx=int(i),
                target_class_idx=int(j),
                source_class_match_embedding=self._get_loop_class_embedding(
                    dataset_idx=source_dataset_idx,
                    class_idx=int(i),
                    include_bootstrap=include_bootstrap,
                ),
                target_class_match_embedding=self._get_loop_class_embedding(
                    dataset_idx=target_dataset_idx,
                    class_idx=int(j),
                    include_bootstrap=include_bootstrap,
                ),
                geometric_distance=float(pairwise_result_matrix[0, i, j]),
                null_distribution_geometric_distance=null_distributions[
                    1:, i, j
                ].tolist(),
                pvalue_permutation=float(pairwise_result_pvalues[i, j]),
                pvalue_corrected_permutation=float(
                    pairwise_result_pvalues_corrected[i, j]
                ),
                t_stats_match=float(matched_t_stats[idx]),
                pvalue_match=float(matched_pvalues[idx]),
                pvalue_corrected_match=float(matched_pvalues_corrected[idx]),
            )
            if dataset_key not in self.loop_matching_result.matches:
                self.loop_matching_result.matches[dataset_key] = []
            else:
                self.loop_matching_result.matches[dataset_key].append(loop_match)
            if verbose:
                logger.info(
                    f"Match found: dataset {source_dataset_idx} class {i} ↔ "
                    f"dataset {target_dataset_idx} class {j} "
                    f"(distance={loop_match.geometric_distance:.4f})"
                )

        if verbose:
            logger.success(
                f"Cross-dataset matching complete: {len(matched_indices)} matches found"
            )
