# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import igraph as ig
import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix, triu

from ..data.base_components import LoopClass
from ..data.boundary import BoundaryMatrixD1
from ..data.constants import (
    DEFAULT_K_YEN,
    DEFAULT_LIFE_PCT,
    DEFAULT_N_COCYCLES_USED,
    DEFAULT_N_FORCE_DEVIATE,
    DEFAULT_N_REPS_PER_LOOP,
)
from ..data.types import Percent_t
from ..data.utils import extract_edges_from_coo, loops_to_coords


def compute_loop_representatives(
    embedding: np.ndarray,
    pairwise_distance_matrix: csr_matrix,
    persistence_diagram: tuple,
    cocycles: list,
    boundary_matrix_d1: BoundaryMatrixD1,
    vertex_ids: list[int],
    top_k: int | None = None,
    n_reps_per_loop: int = DEFAULT_N_REPS_PER_LOOP,
    life_pct: Percent_t = DEFAULT_LIFE_PCT,
    n_cocycles_used: int = DEFAULT_N_COCYCLES_USED,
    n_force_deviate: int = DEFAULT_N_FORCE_DEVIATE,
    k_yen: int = DEFAULT_K_YEN,
    loop_lower_t_pct: float = 2.5,
    loop_upper_t_pct: float = 97.5,
    bootstrap: bool = False,
    rank_offset: int = 0,
) -> list[LoopClass | None]:
    assert pairwise_distance_matrix.shape is not None

    loop_births = np.array(persistence_diagram[0], dtype=np.float32)
    loop_deaths = np.array(persistence_diagram[1], dtype=np.float32)

    if loop_births.size == 0:
        return []
    if top_k is None:
        top_k = loop_births.size
    if top_k <= 0:
        return []
    top_k = min(top_k, loop_births.size)

    persistence = loop_deaths - loop_births
    indices_top_k = np.argsort(persistence)[::-1][:top_k]

    dm_upper = triu(pairwise_distance_matrix, k=1).tocoo()
    edges_array, edge_diameters = extract_edges_from_coo(
        dm_upper.row, dm_upper.col, dm_upper.data
    )

    if len(edges_array) == 0:
        return []

    boundary_edge_set = boundary_matrix_d1.edge_set

    results: list[LoopClass | None] = [None] * len(indices_top_k)

    for i, loop_idx in enumerate(indices_top_k):
        loop_birth = loop_births[loop_idx].item()
        loop_death = loop_deaths[loop_idx].item()

        valid_cocycles = []
        n_cocycles_original = len(cocycles[loop_idx])

        for simplex in cocycles[loop_idx]:
            try:
                verts, coeff = simplex
            except ValueError:
                continue
            if coeff == 0 or len(verts) != 2:
                continue
            u_global = vertex_ids[int(verts[0])]
            v_global = vertex_ids[int(verts[1])]
            edge_global = (min(u_global, v_global), max(u_global, v_global))

            if bootstrap:
                if edge_global in boundary_edge_set or u_global == v_global:
                    valid_cocycles.append(simplex)
            else:
                if edge_global in boundary_edge_set:
                    valid_cocycles.append(simplex)

        if len(valid_cocycles) == 0:
            logger.warning(
                f"Loop class {i + rank_offset}: All {n_cocycles_original} cocycle edges "
                f"filtered (not in boundary matrix). Skipping reconstruction."
            )
            continue

        if len(valid_cocycles) < n_cocycles_original:
            logger.info(
                f"Loop class {i + rank_offset}: Filtered {n_cocycles_original - len(valid_cocycles)}/"
                f"{n_cocycles_original} cocycle edges not in boundary matrix"
            )

        valid_edge_mask = []
        for edge_local in edges_array:
            u_global = vertex_ids[int(edge_local[0])]
            v_global = vertex_ids[int(edge_local[1])]
            edge_global = (min(u_global, v_global), max(u_global, v_global))
            if bootstrap:
                valid = (edge_global in boundary_edge_set) or (u_global == v_global)
            else:
                valid = edge_global in boundary_edge_set
            valid_edge_mask.append(valid)

        valid_edge_mask = np.array(valid_edge_mask, dtype=bool)
        edges_array_filtered = edges_array[valid_edge_mask]
        edge_diameters_filtered = edge_diameters[valid_edge_mask]

        loops_local, _ = reconstruct_n_loop_representatives(
            cocycles_dim1=valid_cocycles,
            edges=edges_array_filtered,
            edge_diameters=edge_diameters_filtered,
            loop_birth=loop_birth,
            loop_death=loop_death,
            n=n_reps_per_loop,
            life_pct=life_pct,
            n_force_deviate=n_force_deviate,
            k_yen=k_yen,
            loop_lower_pct=loop_lower_t_pct,
            loop_upper_pct=loop_upper_t_pct,
            n_cocycles_used=n_cocycles_used,
        )

        loops = [[vertex_ids[v] for v in loop] for loop in loops_local]
        loops_coords = loops_to_coords(embedding=embedding, loops_vertices=loops)

        results[i] = LoopClass(
            rank=i + rank_offset,
            birth=loop_birth,
            death=loop_death,
            cocycles=cocycles[loop_idx],
            representatives=loops,
            coordinates_vertices_representatives=loops_coords,
        )

    return results


def reconstruct_n_loop_representatives(
    cocycles_dim1: List,
    edges: np.ndarray,
    edge_diameters: np.ndarray,
    loop_birth: float,
    loop_death: float,
    n: int,
    life_pct: float = DEFAULT_LIFE_PCT,
    n_force_deviate: int = DEFAULT_N_FORCE_DEVIATE,
    k_yen: int = DEFAULT_K_YEN,
    loop_lower_pct: float = 5,
    loop_upper_pct: float = 95,
    n_cocycles_used: int = DEFAULT_N_COCYCLES_USED,
) -> Tuple[List[List[int]], List[float]]:
    """
    Reconstruct diverse loop representatives
    """
    if n <= 0 or len(edges) == 0:
        return [], []
    filt_t = loop_birth + (loop_death - loop_birth) * life_pct

    all_cocycle_edges: list[tuple[int, int]] = []
    for simplex in cocycles_dim1:
        try:
            verts, coeff = simplex
        except ValueError:
            continue
        if coeff == 0 or len(verts) != 2:
            continue
        all_cocycle_edges.append((int(verts[0]), int(verts[1])))

    if not all_cocycle_edges:
        return [], []

    cocycle_edges_for_paths = all_cocycle_edges[:n_cocycles_used]

    edge_diameters = np.asarray(edge_diameters)
    mask = edge_diameters <= filt_t
    if not np.any(mask):
        return [], []
    edges_filt = edges[mask]
    weights_filt = edge_diameters[mask]

    edge_weight_dict: dict[tuple[int, int], float] = {}
    for i in range(len(edges_filt)):
        key = (
            min(edges_filt[i, 0], edges_filt[i, 1]),
            max(edges_filt[i, 0], edges_filt[i, 1]),
        )
        edge_weight_dict[key] = max(
            edge_weight_dict.get(key, -math.inf), weights_filt[i]
        )

    for e in all_cocycle_edges:
        key = (min(e), max(e))
        edge_weight_dict[key] = math.inf

    cycles_pool: list[list[int]] = []
    cycles_dist: list[float] = []

    for _ in range(n_force_deviate):
        edge_list = list(edge_weight_dict.keys())
        weight_list = [edge_weight_dict[e] for e in edge_list]

        if not edge_list:
            break

        n_vertices = max(max(e) for e in edge_list) + 1
        g = ig.Graph(n=n_vertices, edges=edge_list, directed=False)
        g.es["weight"] = weight_list

        paths_this_round: list[list[int]] = []
        for i, j in cocycle_edges_for_paths:
            paths = _k_shortest_paths(g, i, j, k_yen)
            if not paths:
                continue
            for path in paths:
                dist = _path_weight(g, path)
                if math.isfinite(dist):
                    cycles_pool.append(path)
                    paths_this_round.append(path)
                    cycles_dist.append(dist)

        for path in paths_this_round:
            for u, v in zip(path[:-1], path[1:]):
                key = (min(u, v), max(u, v))
                edge_weight_dict[key] = math.inf

    return _select_diverse_loops(
        cycles=cycles_pool,
        distances=cycles_dist,
        n=n,
        lower_pct=loop_lower_pct,
        upper_pct=loop_upper_pct,
    )


def _k_shortest_paths(g: ig.Graph, source: int, target: int, k: int) -> list[list[int]]:
    if source == target:
        return []
    try:
        return g.get_k_shortest_paths(
            source, target, k=k, weights=g.es["weight"], mode="ALL"
        )
    except ig._igraph.InternalError:
        return []


def _path_weight(g: ig.Graph, path: Sequence[int]) -> float:
    if len(path) < 2:
        return math.inf
    weight = 0.0
    for u, v in zip(path[:-1], path[1:]):
        try:
            eid = g.get_eid(u, v, directed=False)
        except ig._igraph.InternalError:
            return math.inf
        w = g.es[eid]["weight"]
        if not math.isfinite(w):
            return math.inf
        weight += float(w)
    return weight


def _select_diverse_loops(
    cycles: Iterable[Sequence[int]],
    distances: Iterable[float],
    n: int,
    lower_pct: float,
    upper_pct: float,
) -> Tuple[List[List[int]], List[float]]:
    pairs = sorted(
        [(float(d), list(c)) for d, c in zip(distances, cycles) if math.isfinite(d)],
        key=lambda x: x[0],
    )
    if not pairs:
        return [], []

    n_total = len(pairs)
    n_return = min(n_total, n)
    if n_return == 1:
        idxs = [n_total // 2]
    else:
        step = (upper_pct - lower_pct) / (n_return - 1)
        idxs = []
        for i in range(n_return):
            pct = (lower_pct + step * i) / 100
            idx = min(int(math.floor(n_total * pct)), n_total - 1)
            idxs.append(idx)

    selected = [pairs[i] for i in idxs]
    dists = [p[0] for p in selected]
    loops = [p[1] for p in selected]
    return loops, dists
