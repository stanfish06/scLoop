# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Mchigan)
from __future__ import annotations

import ast
import pickle
import uuid
from dataclasses import dataclass, field
from itertools import chain, combinations, product
from typing import Literal

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist, directed_hausdorff, pdist, squareform
from scipy.stats import binom, false_discovery_control, gamma

from .utils import (
    compute_homological_equivalence,
    edge_idx_encode,
    trajectory_distance,
)


# TODO: modify this data sturcture to enable cross species matching
# TODO: extract developmetnal trajectories from loop representatives (thoughts: embed pseudotime gradient with hodge decomposition then split the loop in the case of lineage convergence)
# EdgeCollapose Rips does not work as it will heavilly reduce local connectivity
# random downsample for large input? use topological aware subsampling
# wrap ripser.cpp instead? I technically only need cocylces
# random downsample for large input?
# persistance landscape, useful here or no?
@dataclass
class HomologyData:
    data: np.ndarray
    data_visualization: np.ndarray = field(default_factory=lambda: np.array([]))
    data_cross_match: np.ndarray = field(default_factory=lambda: np.array([]))
    n_vertices: int = 0
    pseudotime: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # pseudotime values for each point
    persistence_diagram: np.ndarray = field(default_factory=lambda: np.array([]))
    loops_eidx: list[list[np.ndarray]] = field(default_factory=list)
    loops_coords: list[list[np.ndarray]] = field(default_factory=list)
    loops_coords_visualization: list[list[np.ndarray]] = field(default_factory=list)
    bd_mat: tuple = ()
    bd_column_birth_t: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    bd_row_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    bd_mat_0: tuple = ()  # boundary matrix from edges to vertices
    edge_pseudotime_deltas: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # pseudotime differences for edges
    loop_edge_clusters: list[list[dict]] = field(
        default_factory=list
    )  # clustering results for each loop
    persistence_diagram_boot: list[np.ndarray] = field(default_factory=list)
    loops_eidx_boot: list[list[list[np.ndarray]]] = field(default_factory=list)
    loops_coords_boot: list[list[list[np.ndarray]]] = field(default_factory=list)
    loops_coords_visualization_boot: list[list[list[np.ndarray]]] = field(
        default_factory=list
    )
    matching_df: list[pd.DataFrame] = field(default_factory=list)
    tracks: dict = field(default_factory=dict)  # e.g. (1,2): [(2,3),...]
    tracks_pvals: dict = field(default_factory=dict)
    n_booted: int = 0
    loop_rank: pd.DataFrame = field(default_factory=pd.DataFrame)
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.data_visualization) != len(self.data):
            self.data_visualization = self.data
        self.n_vertices = self.data.shape[0]
        self.dist_mat = pdist(self.data)
        self.dummy_value = np.min(self.dist_mat[self.dist_mat > 0]) * 1e-3
        self.dist_mat = squareform(self.dist_mat)
        self.dist_mat += self.dummy_value
        np.fill_diagonal(self.dist_mat, 0.0)

        # Validate pseudotime if provided
        if len(self.pseudotime) > 0:
            if len(self.pseudotime) != self.n_vertices:
                raise ValueError(
                    f"Pseudotime length ({len(self.pseudotime)}) must match number of vertices ({self.n_vertices})"
                )

    def compute_homology(self, thresh=None):
        filt = julia.Rips(self.dist_mat, sparse=True, threshold=thresh)
        # don't compute representatives if only persistence diagram is needed
        result_cycle = julia.ripserer(filt, reps=False)
        birth_t = np.array([i[1] for i in result_cycle[1]])
        death_t = np.array([i[2] for i in result_cycle[1]])
        # order from most persistent to least persistent
        self.persistence_diagram = np.vstack([birth_t, death_t]).T[::-1, :]
        self.parameters["filtration_threshold_homology"] = thresh

    def compute_boundary_matrix(self, thresh=None):
        if thresh is None:
            if (
                "filtration_threshold_homology" in self.parameters
                and self.parameters["filtration_threshold_homology"]
            ):
                thresh = self.parameters["filtration_threshold_homology"]
        filt = julia.Rips(self.dist_mat, sparse=True, threshold=thresh)
        trigs, birth_t = julia.boundary_mat_d2(filt)
        trigs = np.array(np.array(trigs).tolist()) - 1
        edges = [list(combinations(trig, 2)) for trig in trigs]
        birth_t = np.array(birth_t)

        edge_idx = np.concatenate(
            [[edge_idx_encode(i, j, self.n_vertices) for i, j in es] for es in edges]
        )
        self.bd_row_id = np.unique(edge_idx)
        # boundary matrix from triangles to edges
        one_ridx_A = np.searchsorted(self.bd_row_id, edge_idx).astype(int)
        nrow_A = len(self.bd_row_id)
        ncol_A = int(len(one_ridx_A) / 3)
        one_cidx_A = np.repeat(np.arange(ncol_A), 3).astype(int)

        # boundary matrix from edges to vertices
        A0 = np.array(np.concatenate(edges))
        # Deduplicate edges based on encoded edge_idx
        _, uniq_idx = np.unique(edge_idx, return_index=True)
        A0_unique = A0[uniq_idx]  # One row per unique edge
        vids = np.unique(A0_unique.flatten())

        # Map edge indices for unique edges
        edge_ridx = np.searchsorted(self.bd_row_id, edge_idx[uniq_idx])

        # Create boundary matrix: 2 entries per unique edge (one for each vertex)
        v1 = np.stack([A0_unique[:, 0], edge_ridx], 1)
        v2 = np.stack([A0_unique[:, 1], edge_ridx], 1)
        # do so such that two vertices of each edge are adjacent
        A0 = np.empty((A0_unique.shape[0] * 2, 2), dtype=int)
        A0[0::2] = v1
        A0[1::2] = v2
        A0[:, 0] = np.searchsorted(vids, A0[:, 0]).astype(int)

        self.bd_column_birth_t = birth_t
        self.bd_mat = (one_ridx_A, one_cidx_A, nrow_A, ncol_A)
        self.bd_mat_0 = (A0[:, 0], A0[:, 1], len(vids), nrow_A)
        self.parameters["filtration_threshold_bd_matrix"] = thresh

    def compute_loop_representatives(
        self,
        n_top,
        n_each=4,
        life_pct=0.05,
        n_force_deviate=4,
        n_reps_per_loop=4,
        loop_lower_pct=5,
        loop_upper_pct=95,
        n_max_cocycles=1,
    ):
        assert "filtration_threshold_homology" in self.parameters, (
            "run compute_homology first"
        )
        filt = julia.Rips(
            self.dist_mat,
            sparse=True,
            threshold=self.parameters["filtration_threshold_homology"],
        )
        cocycles = julia.ripserer(filt, reps=1)
        n_total_loops = len(cocycles[1])
        n_compute = min(n_total_loops, n_top)
        self.loops_coords = []
        self.loops_coords_visualization = []
        self.loops_eidx = []
        if n_compute > 0:
            for i in range(n_compute):
                reps = julia.reconstruct_n_loop_representatives(
                    cocycles,
                    filt,
                    i,
                    n_each,
                    life_pct,
                    n_force_deviate,
                    n_reps_per_loop,
                    loop_lower_pct,
                    loop_upper_pct,
                    n_max_cocycles,
                )
                # julia to python
                reps = [list(lp) for lp in reps[0]]
                reps_eidx = []
                reps_coords = []
                reps_coords_visualization = []
                for k in range(len(reps)):
                    rep_i_idx = [j - 1 for j in reps[k]]
                    rep_i_idx.append(rep_i_idx[0])
                    rep_i_coords = []
                    rep_i_coords_visualization = []
                    rep_i_eidx = []

                    for j in range(1, len(rep_i_idx)):
                        v1 = rep_i_idx[j - 1]
                        v2 = rep_i_idx[j]
                        edge_idx = edge_idx_encode(v1, v2, self.n_vertices)
                        if edge_idx in self.bd_row_id:
                            rep_i_eidx.append(
                                np.where(edge_idx == self.bd_row_id)[0][0]
                            )
                        rep_i_coords.append(self.data[v1, :])
                        rep_i_coords_visualization.append(
                            self.data_visualization[v1, :]
                        )
                    # circles back to origin
                    edge_idx = edge_idx_encode(
                        i=v2, j=rep_i_idx[0], n_vertices=self.n_vertices
                    )
                    if edge_idx in self.bd_row_id:
                        rep_i_eidx.append(np.where(edge_idx == self.bd_row_id)[0][0])
                    rep_i_coords.append(self.data[rep_i_idx[0], :])
                    rep_i_coords_visualization.append(
                        self.data_visualization[rep_i_idx[0], :]
                    )

                    reps_eidx.append(np.array(rep_i_eidx))
                    reps_coords.append(np.array(rep_i_coords))
                    reps_coords_visualization.append(
                        np.array(rep_i_coords_visualization)
                    )
                self.loops_coords.append(reps_coords)
                self.loops_coords_visualization.append(reps_coords_visualization)
                self.loops_eidx.append(reps_eidx)

            for i in range(len(self.loops_eidx)):
                if f"(0,{i})" not in self.tracks:
                    self.tracks[f"(0,{i})"] = {
                        "birth_t": self.persistence_diagram[i, 0],
                        "death_t": self.persistence_diagram[i, 1],
                        "loops": [(0, i)],
                    }

    def _compute_edge_pseudotime_deltas(self):
        if len(self.pseudotime) == 0:
            raise ValueError("Pseudotime values not provided")
        if not self.bd_mat_0:
            raise ValueError("Boundary matrix bd_mat_0 not computed")
        one_ridx_A0, one_cidx_A0, nrow_A0, ncol_A0 = self.bd_mat_0
        # Initialize edge pseudotime deltas
        self.edge_pseudotime_deltas = np.zeros(len(self.bd_row_id))
        # bd_mat_0 has 2 entries per edge (one for each vertex)
        for edge_idx in range(len(self.bd_row_id)):
            # Find the two vertices for this edge
            vertex_mask = one_cidx_A0 == edge_idx
            vertices = one_ridx_A0[vertex_mask]

            if len(vertices) == 2:
                v1, v2 = vertices[0], vertices[1]
                self.edge_pseudotime_deltas[edge_idx] = (
                    self.pseudotime[v1] - self.pseudotime[v2]
                )

    # thresh_t should be picked such that the target loop actually exists
    def compute_hodge_laplacian(self, thresh_t, normalized=True):
        if not self.bd_mat:
            raise ValueError("compute boundary matrix first")
        # Extract boundary matrices
        # ∂₂: triangles → edges (bd_mat)
        one_ridx_A, one_cidx_A, nrow_A, ncol_A = self.bd_mat
        # remove columns with birth t larger than the death t of the loop
        columns_use = np.where(self.bd_column_birth_t <= thresh_t)[0]
        if columns_use.size == 0:
            return None
        mask = np.isin(one_cidx_A, columns_use)
        one_ridx_A_local = one_ridx_A[mask]
        one_cidx_A_local = one_cidx_A[mask]
        _, one_cidx_A_local = np.unique(one_cidx_A_local, return_inverse=True)
        ncol_A_local = np.max(one_cidx_A_local) + 1
        # Create sparse matrix with proper orientation (±1)
        data_A = np.ones(len(one_ridx_A_local))
        # make boundary matrix oriented (the edge encoding already sorted the edges by vertex index)
        # the second edge (relative) of each triangle will be -1
        data_A[np.mod(np.arange(len(one_ridx_A_local)), 3) == 1] = -1
        bd2 = csr_matrix(
            (data_A, (one_ridx_A_local, one_cidx_A_local)), shape=(nrow_A, ncol_A_local)
        )
        # ∂₁: edges → vertices (bd_mat_0)
        one_ridx_A0, one_cidx_A0, nrow_A0, ncol_A0 = self.bd_mat_0
        # Create sparse matrix with proper orientation
        # for each edge, the second (relative) vertex gets -1
        data_A0 = np.ones(len(one_ridx_A0))
        data_A0[np.mod(np.arange(len(one_ridx_A0)), 2) == 1] = -1
        bd1 = csr_matrix(
            (data_A0, (one_ridx_A0, one_cidx_A0)), shape=(nrow_A0, ncol_A0)
        )
        if normalized:
            D2 = np.maximum(abs(bd2).sum(1), 1)
            D1 = 2 * (abs(bd1) @ D2)
            D3 = 1 / 3
            # L1 = D2 B1t D1^(-1) B1 + B2 D3 B2t D2^(-1)
            L1 = (bd1.T.multiply(D2).multiply(1 / D1.T)) @ bd1 + (bd2 @ bd2.T).multiply(
                1 / D2.T
            ) * D3
        else:
            L1 = bd1.T @ bd1 + bd2 @ bd2.T
        # Compute pseudotime deltas for edges if pseudotime is provided
        if len(self.pseudotime) > 0:
            self._compute_edge_pseudotime_deltas()
        return L1

    # thresh_t should be picked such that the target loop actually exists
    def compute_hodge_eigendecomposition(
        self, thresh_t, n_components=10, normalized=True
    ):
        L1 = self.compute_hodge_laplacian(normalized=normalized, thresh_t=thresh_t)
        if L1 is None:
            raise ValueError("Hodge Laplacian undefined")
        # Compute smallest eigenvalues and corresponding eigenvectors
        n_comp = min(n_components, L1.shape[0] - 2)
        eigenvalues, eigenvectors = eigsh(L1, k=n_comp, which="SM")
        # Sort by eigenvalue
        sort_idx = np.argsort(eigenvalues)
        hodge_eigenvalues = eigenvalues[sort_idx]
        hodge_eigenvectors = eigenvectors[:, sort_idx]
        return hodge_eigenvalues, hodge_eigenvectors

    def computing_loop_edge_embedding(
        self, loop_idx, life_pct=0.05, n_hodge_components=10, edge_half_window=1
    ):
        if f"(0,{loop_idx})" not in self.tracks:
            raise ValueError("No representatives found for this loop ")
        loop_idx = f"(0,{loop_idx})"
        birth_t = self.tracks[loop_idx]["birth_t"]
        death_t = self.tracks[loop_idx]["death_t"]
        thresh_t = birth_t + (death_t - birth_t) * life_pct
        eigenvalues, eigenvectors = self.compute_hodge_eigendecomposition(
            thresh_t=thresh_t, n_components=n_hodge_components
        )
        self.tracks[loop_idx]["hodge_eigenvalues"] = eigenvalues
        self.tracks[loop_idx]["hodge_eigenvectors"] = eigenvectors
        self.tracks[loop_idx]["loops_edges_embedding"] = []

        # Count total loops for progress tracking
        total_loops = sum(
            len(self.loops_eidx[rid])
            if sid == 0
            else len(self.loops_eidx_boot[sid - 1][rid])
            for sid, rid in self.tracks[loop_idx]["loops"]
            if (
                self.loops_eidx[rid] if sid == 0 else self.loops_eidx_boot[sid - 1][rid]
            )
        )

        with Progress(
            TextColumn("[bold blue]Computing edge embeddings"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[bold green]loops"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing loops", total=total_loops)

            for sid, rid in self.tracks[loop_idx]["loops"]:
                if sid == 0:
                    loops_eidx = self.loops_eidx[rid]
                else:
                    loops_eidx = self.loops_eidx_boot[sid - 1][rid]
                loops_edges_embed = []
                if len(loops_eidx) > 0:
                    for lp in loops_eidx:
                        edge_mat = np.zeros([len(self.bd_row_id), len(lp)])
                        for i in range(len(lp)):
                            edge_window = np.take(
                                lp,
                                np.arange(
                                    i - edge_half_window, i + edge_half_window + 1
                                ),
                                mode="wrap",
                            )
                            edge_mat[edge_window, i] = self.edge_pseudotime_deltas[
                                edge_window
                            ]
                        edges_embed = edge_mat.T @ eigenvectors
                        loops_edges_embed.append(edges_embed)
                        progress.advance(task)
                self.tracks[loop_idx]["loops_edges_embedding"].append(loops_edges_embed)

    # after booting both datasets, match two groups of loops and use permutation test to match groups
    # TODO: use numba to speed up cross matching
    def cross_match(self, reference: HomologyData, n_permute: int) -> pd.DataFrame:
        assert self.tracks, "Query data contains no bootstrapped samples"
        assert reference.tracks, "Reference data contains no bootstrapped samples"
        import rpy2.robjects as ro

        similarity_func = SimilarityMeasures_helper(ro)

        # frechet distance is too slow to compute
        # TODO: implement frechet distance
        def compute_loop_distance(
            l1: np.ndarray, l2: np.ndarray, type: str = "hausdorff"
        ):
            if type == "frechet":
                dist_mat = cdist(l1, l2)
                p, q = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
                try:
                    d1 = trajectory_distance(
                        np.roll(l1, -p, axis=0),
                        np.roll(l2, -q, axis=0),
                        "Frechet",
                        similarity_func,
                    )[0]
                    d2 = trajectory_distance(
                        np.roll(l1, -p, axis=0),
                        np.roll(l2[::-1], q + 1, axis=0),
                        "Frechet",
                        similarity_func,
                    )[0]
                    # not sure what negative value means
                    if d1 < 0:
                        d1 = np.inf
                    if d2 < 0:
                        d2 = np.inf
                    dist = min(d1, d2)
                except ValueError:
                    dist = np.nan
            elif type == "hausdorff":
                dist = max(directed_hausdorff(l1, l2)[0], directed_hausdorff(l2, l1)[0])
            return dist

        def group_distance(g1: list[np.ndarray], g2: list[np.ndarray]):
            g1_pairs = combinations(range(len(g1)), 2)
            g2_pairs = combinations(range(len(g2)), 2)
            g1_g2_pairs = product(range(len(g1)), range(len(g2)))
            g1_in_dist = np.nanmean(
                np.array(
                    [compute_loop_distance(l1=g1[i], l2=g1[j]) for i, j in g1_pairs]
                )
            )
            g2_in_dist = np.nanmean(
                np.array(
                    [compute_loop_distance(l1=g2[i], l2=g2[j]) for i, j in g2_pairs]
                )
            )
            g1_g2_dist = np.nanmean(
                np.array(
                    [compute_loop_distance(l1=g1[i], l2=g2[j]) for i, j in g1_g2_pairs]
                )
            )
            return (g1_g2_dist) / ((g1_in_dist + g2_in_dist) / 2)

        stats_permuate = []
        with Progress(refresh_per_second=25) as progress:
            task_cross_compare = progress.add_task(
                "[bold blue]Cross-dataset comparison",
                total=len(self.tracks) * len(reference.tracks),
            )
            task_permute = progress.add_task(
                "[green]  Permutation testing", total=n_permute, visible=False
            )
            cross_pairs = list(product(self.tracks.keys(), reference.tracks.keys()))
            result = []
            for pair_idx, (i, j) in enumerate(cross_pairs):
                progress.update(
                    task_cross_compare,
                    description=f"[bold blue]Cross-dataset comparison - Pair {pair_idx + 1}/{len(cross_pairs)} ({i} vs {j})",
                )
                query_tracks = self.tracks[i]["loops"]
                reference_tracks = reference.tracks[j]["loops"]
                query_loops_coords = list(
                    chain.from_iterable(
                        [
                            self.loops_coords[tid[1]]
                            if tid[0] == 0
                            else self.loops_coords_boot[tid[0] - 1][tid[1]]
                            for tid in query_tracks
                        ]
                    )
                )
                reference_loops_coords = list(
                    chain.from_iterable(
                        [
                            reference.loops_coords[tid[1]]
                            if tid[0] == 0
                            else reference.loops_coords_boot[tid[0] - 1][tid[1]]
                            for tid in reference_tracks
                        ]
                    )
                )
                stats_test = group_distance(query_loops_coords, reference_loops_coords)
                combined_loops_coords = query_loops_coords + reference_loops_coords
                stats_permute = []
                # Show permutation task
                progress.update(task_permute, visible=True)
                progress.reset(task_permute, total=n_permute)
                progress.update(
                    task_permute,
                    description=f"[green]  Permutation testing - {n_permute} iterations",
                )
                for perm_idx in range(n_permute):
                    reorder_idx = np.random.permutation(len(combined_loops_coords))
                    combined_loops_coords_reorder = [
                        combined_loops_coords[i] for i in reorder_idx
                    ]
                    stats_permute.append(
                        group_distance(
                            combined_loops_coords_reorder[: len(query_loops_coords)],
                            combined_loops_coords_reorder[len(query_loops_coords) :],
                        )
                    )
                    progress.update(task_permute, advance=1)
                progress.update(task_permute, visible=False)
                stats_permute = np.array(stats_permute)
                # do one sided test
                pval = np.sum(stats_permute >= stats_test) / len(stats_permute)
                # pair, test statistics, pval, permuted statistics
                result.append(((i, j), stats_test, pval, stats_permute))
                progress.update(
                    task_cross_compare,
                    advance=1,
                )
            df = pd.DataFrame(
                result,
                columns=pd.Index(
                    ["pair", "frechet_test", "frechet_pval", "frechet_permute"]
                ),
            )
            return df

    def clean_boot(self):
        self.tracks = {}
        self.n_booted = 0
        self.loops_coords_boot = []
        self.loops_coords_visualization_boot = []
        self.loops_eidx_boot = []
        self.persistence_diagram_boot = []
        self.matching_df = []

    def write_pkl(self, fname=None):
        if fname is not None:
            with open(f"{fname}.pkl", "wb") as f:
                pickle.dump(self, f)
        else:
            fname = str(uuid.uuid4().hex)
            with open(f"{fname}.pkl", "wb") as f:
                pickle.dump(self, f)

    # Thoughts: for approx, use permutation test to have better match, but this will be computationally intense for sure
    # TODO: comme up with a reasonable way for loop matching when doing approximation
    # TODO: transfer parameter values from compute_loop_representatives to here
    def boot(
        self,
        n,
        thresh=None,
        max_geometric_dist=np.inf,
        max_homological_dist=np.inf,
        n_reps_per_loop=4,
        rep_life_pct=0.1,
        n_nearest_loops=20,
        regression_mode: Literal["exact", "approx"] = "exact",
        ridge_coef_a=0.1,
        ridge_coef_b=1,
        do_approximation=True,
        n_neighbors=1,
        fresh_start=True,
        n_force_deviate=4,
        _n_reps_per_loop=4,
        loop_lower_pct=5,
        loop_upper_pct=95,
        n_max_cocycles=1,
        do_downsample=True,
        fraction_downsample=0,
    ):
        if not self.bd_mat:
            raise ValueError("compute boundary matrix first")
        if not self.loops_eidx:
            raise ValueError("compute original loops first")
        if fresh_start:
            self.clean_boot()
        if thresh is None:
            if (
                "filtration_threshold_homology" in self.parameters
                and self.parameters["filtration_threshold_homology"]
            ):
                thresh = self.parameters["filtration_threshold_homology"]
        source_loop_eidx_pool = self.loops_eidx.copy()
        source_loop_coords_pool = self.loops_coords.copy()
        source_loop_key = []
        source_loop_birth_t = []
        source_loop_death_t = []
        for i in range(len(self.loops_eidx)):
            sloop_birth_t = self.persistence_diagram[i, 0]
            sloop_death_t = self.persistence_diagram[i, 1]
            if f"(0,{i})" not in self.tracks:
                self.tracks[f"(0,{i})"] = {
                    "birth_t": sloop_birth_t,
                    "death_t": sloop_death_t,
                    "loops": [(0, i)],
                }
            source_loop_key.append(f"(0,{i})")
            source_loop_birth_t.append(sloop_birth_t)
            source_loop_death_t.append(sloop_death_t)
        # if there are tracks/heads in tracks, send them to source loop
        if self.tracks:
            for sid in self.tracks.keys():
                # boot idx start from 1, 0 is the original batch
                batch_id, loop_id = ast.literal_eval(sid)
                if batch_id > 0:
                    source_loop_eidx_pool.append(
                        self.loops_eidx_boot[batch_id - 1][loop_id]
                    )
                    source_loop_coords_pool.append(
                        self.loops_coords_boot[batch_id - 1][loop_id]
                    )
                    source_loop_key.append(sid)
        # edge lookup
        edge_lookup = {edge_id: idx for idx, edge_id in enumerate(self.bd_row_id)}
        # Pre-convert to numpy array for faster access
        source_loop_death_t = np.array(source_loop_death_t)
        # Cache boundary matrix components
        bd_nrow_A = int(self.bd_mat[2])
        bd_ncol_A = int(self.bd_mat[3])
        # Custom progress columns
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ]
        with Progress(*progress_columns, refresh_per_second=10) as progress:
            # Main bootstrap progress
            task_boot = progress.add_task("[bold cyan]Bootstrap Progress", total=n)
            # Sub-tasks (initially hidden)
            task_find_loop = progress.add_task(
                "[green]  Finding loops", total=1, visible=False
            )
            task_geometric_dist = progress.add_task(
                "[yellow]  Computing distances", total=1, visible=False
            )
            task_homology = progress.add_task(
                "[magenta]  Homological analysis", total=1, visible=False
            )
            n_success = 0
            for boot_iter in range(n):
                # Update main progress with stats
                success_rate = (
                    (n_success / (boot_iter + 1)) * 100 if boot_iter > 0 else 0
                )
                progress.update(
                    task_boot,
                    description=f"[bold cyan]Bootstrap Progress - Round {boot_iter + 1}/{n} - Success: {n_success} ({success_rate:.1f}%)",
                )
                boot_idx = np.random.choice(
                    self.data.shape[0],
                    size=self.data.shape[0],
                    replace=True,
                )
                x_boot = self.data[boot_idx]
                x_boot = x_boot + np.random.normal(
                    scale=self.dummy_value, size=self.data.shape
                )
                dist_mat = squareform(pdist(x_boot))
                dist_mat += self.dummy_value
                np.fill_diagonal(dist_mat, 0.0)
                filt = julia.Rips(dist_mat, sparse=True, threshold=thresh)
                cocycles = julia.ripserer(filt, reps=1)
                birth_t = np.array([i[1] for i in cocycles[1]])[::-1]
                death_t = np.array([i[2] for i in cocycles[1]])[::-1]
                reps_eidx_boot = []
                reps_coord_boot = []
                reps_coord_visualization_boot = []
                # Show and configure loop finding task
                progress.update(task_find_loop, visible=True)
                progress.reset(task_find_loop, total=len(cocycles[1]))
                progress.update(
                    task_find_loop,
                    description=f"[green]  Finding loops - Processing {len(cocycles[1])} cocycles",
                )
                for nc in range(len(cocycles[1])):
                    try:
                        reps = julia.reconstruct_n_loop_representatives(
                            cocycles,
                            filt,
                            nc,
                            n_reps_per_loop,
                            rep_life_pct,
                            n_force_deviate,
                            _n_reps_per_loop,
                            loop_lower_pct,
                            loop_upper_pct,
                            n_max_cocycles,
                        )
                        reps = [list(lp) for lp in reps[0]]
                        reps_eidx = []
                        reps_coords = []
                        reps_coords_visualization = []
                        for i in range(len(reps)):
                            rep_i_idx = [j - 1 for j in reps[i]]
                            rep_i_idx.append(rep_i_idx[0])
                            rep_i_coords = []
                            rep_i_coords_visualization = []
                            rep_i_eidx = []

                            for j in range(1, len(rep_i_idx)):
                                v1 = boot_idx[rep_i_idx[j - 1]]
                                v2 = boot_idx[rep_i_idx[j]]
                                edge_idx = edge_idx_encode(v1, v2, self.n_vertices)
                                if edge_idx in edge_lookup:
                                    rep_i_eidx.append(edge_lookup[edge_idx])
                                rep_i_coords.append(self.data[v1, :])
                                rep_i_coords_visualization.append(
                                    self.data_visualization[v1, :]
                                )
                            reps_eidx.append(np.array(rep_i_eidx))
                            reps_coords.append(np.array(rep_i_coords))
                            reps_coords_visualization.append(
                                np.array(rep_i_coords_visualization)
                            )
                        reps_eidx_boot.append(reps_eidx)
                        reps_coord_boot.append(reps_coords)
                        reps_coord_visualization_boot.append(reps_coords_visualization)
                    except ValueError:
                        continue
                    progress.update(task_find_loop, advance=1)
                # Hide completed task
                progress.update(task_find_loop, visible=False)
                # Skip if no loops found
                if len(reps_eidx_boot) == 0:
                    continue
                n_source_groups = len(source_loop_eidx_pool)
                n_boot_groups = len(reps_eidx_boot)
                total_comparisons = n_source_groups * n_boot_groups
                # Show and configure distance task
                progress.update(task_geometric_dist, visible=True)
                progress.reset(task_geometric_dist, total=total_comparisons)
                progress.update(
                    task_geometric_dist,
                    description=f"[yellow]  Computing distances - {n_source_groups} x {n_boot_groups} comparisons",
                )
                geometric_distances = np.full((n_source_groups, n_boot_groups), np.nan)
                for i in range(n_source_groups):
                    for j in range(n_boot_groups):
                        n_source_loops = len(source_loop_coords_pool[i])
                        n_target_loops = len(reps_coord_boot[j])
                        dists = []
                        for ii in range(n_source_loops):
                            for jj in range(n_target_loops):
                                l1 = source_loop_coords_pool[i][ii]
                                l2 = reps_coord_boot[j][jj]
                                dist = np.nan
                                try:
                                    dist = max(
                                        directed_hausdorff(l1, l2)[0],
                                        directed_hausdorff(l2, l1)[0],
                                    )
                                finally:
                                    dists.append(dist)
                        geometric_distances[i, j] = np.nanmean(dists)
                        progress.update(task_geometric_dist, advance=1)
                progress.update(task_geometric_dist, visible=False)
                # Filter pairs
                pairs_filt = []
                for si in range(n_source_groups):
                    distances = geometric_distances[si, :]
                    valid_mask = ~np.isnan(distances)

                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        valid_distances = distances[valid_mask]

                        n_keep = min(len(valid_distances), n_nearest_loops)
                        if n_keep == len(valid_distances):
                            top_indices = np.arange(len(valid_distances))
                        else:
                            top_indices = np.argpartition(valid_distances, n_keep - 1)[
                                :n_keep
                            ]

                        for idx in top_indices:
                            j = valid_indices[idx]
                            pairs_filt.append(
                                (
                                    ((si, j), source_loop_death_t[si]),
                                    valid_distances[idx],
                                )
                            )
                if pairs_filt:
                    # Show homology task
                    progress.update(task_homology, visible=True)
                    progress.reset(task_homology, total=len(pairs_filt))
                    progress.update(
                        task_homology,
                        description=f"[magenta]  Homological analysis - {len(pairs_filt)} pairs",
                    )
                    result = []
                    for idx, (((i, j), sloop_death_t), _) in enumerate(pairs_filt):
                        result.append(
                            compute_homological_equivalence(
                                source_loops_edges=source_loop_eidx_pool[i],
                                target_loops_edges=reps_eidx_boot[j],
                                one_ridx_A=self.bd_mat[0],
                                one_cidx_A=self.bd_mat[1],
                                nrow_A=bd_nrow_A,
                                ncol_A=bd_ncol_A,
                                ridge_coef_a=ridge_coef_a,
                                ridge_coef_b=ridge_coef_b,
                                do_approximation=do_approximation,
                                n_neighbors=n_neighbors,
                                bd_column_birth_t=self.bd_column_birth_t,
                                source_loop_death_t=sloop_death_t,
                                regression_mode=regression_mode,
                                do_downsample=do_downsample,
                                fraction_downsample=fraction_downsample,
                            )
                        )
                        progress.update(task_homology, advance=1)
                    progress.update(task_homology, visible=False)

                    df = pd.DataFrame(
                        pairs_filt, columns=pd.Index(["pair", "geometric_dist"])
                    )
                    df["homological_dist"] = result
                    self.matching_df.append(df.copy())
                    df = df[
                        np.logical_and(
                            df["homological_dist"] < max_homological_dist,
                            ~np.isnan(df["homological_dist"]),
                        )
                    ]
                    if df.empty:
                        continue
                    self.loops_eidx_boot.append(reps_eidx_boot)
                    self.loops_coords_boot.append(reps_coord_boot)
                    self.loops_coords_visualization_boot.append(
                        reps_coord_visualization_boot
                    )
                    self.persistence_diagram_boot.append(
                        np.vstack([birth_t, death_t]).T
                    )
                    df[["source", "target"]] = np.array([p for p, _ in df["pair"]])
                    if regression_mode == "exact":
                        df = df.loc[df["homological_dist"] < 1, :]
                        df["cost"] = df["geometric_dist"]
                    else:
                        df["cost"] = df["homological_dist"] * df["geometric_dist"]

                    cost_matrix_sub = df.pivot(
                        index="source", columns="target", values="cost"
                    ).fillna(np.inf)
                    cost_matrix = np.full(
                        [
                            cost_matrix_sub.shape[0],
                            cost_matrix_sub.shape[0] + cost_matrix_sub.shape[1],
                        ],
                        max_homological_dist * max_geometric_dist,
                    )
                    cost_matrix[
                        : cost_matrix_sub.shape[0], : cost_matrix_sub.shape[1]
                    ] = cost_matrix_sub
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    mask = col_ind >= cost_matrix_sub.shape[1]
                    row_ind = cost_matrix_sub.index[row_ind[~mask]]
                    col_ind = cost_matrix_sub.columns[col_ind[~mask]]
                    for j in range(len(row_ind)):
                        skey = source_loop_key[row_ind[j]]
                        self.tracks[skey]["loops"].append(
                            (self.n_booted + 1, col_ind[j])
                        )
                    n_success += 1
                    self.n_booted += 1
                progress.update(task_boot, advance=1)

    def rank_loops(self):
        if not self.n_booted:
            raise ValueError("do bootstrapping first")
        presence_probs = [
            (
                ast.literal_eval(src_loop),
                len(track["loops"]),
                (len(track["loops"]) - 1) / self.n_booted,
            )
            for src_loop, track in self.tracks.items()
        ]
        mean_presence_prob = np.mean([p for _, _, p in presence_probs])
        presence_pvals = [
            (src_loop, p, 1 - binom.cdf(n, self.n_booted, mean_presence_prob))
            for src_loop, n, p in presence_probs
        ]
        # null distribution for lifetime
        lifetimes = [self.persistence_diagram[:, 1] - self.persistence_diagram[:, 0]]
        for boot_ph in self.persistence_diagram_boot:
            lifetimes.append(boot_ph[:, 1] - boot_ph[:, 0])
        lifetimes_full = np.concatenate(lifetimes)
        lifetimes_full = lifetimes_full[lifetimes_full < np.inf]
        params = gamma.fit(lifetimes_full, floc=0)
        persistence_pvals = [
            (
                lifetimes[src_loop[0]][src_loop[1]],
                1
                - gamma.cdf(
                    lifetimes[src_loop[0]][src_loop[1]],
                    params[0],
                    loc=params[1],
                    scale=params[2],
                ),
            )
            for src_loop, _, _ in presence_probs
        ]
        self.parameters["gamma_fit"] = params
        self.parameters["binom_fit"] = mean_presence_prob
        df = pd.DataFrame(
            presence_pvals, columns=["src_loop", "prob_presence", "pval_presence"]
        )
        df[["src_lifetime", "pval_persistence"]] = persistence_pvals
        df["pval_presence_adjust"] = false_discovery_control(df["pval_presence"])
        df["pval_persistence_adjust"] = false_discovery_control(df["pval_persistence"])
        self.loop_rank = df
