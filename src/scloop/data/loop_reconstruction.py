# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import igraph as ig
import numpy as np


def reconstruct_n_loop_representatives(
    cocycles_dim1: List,
    edges: np.ndarray,
    edge_diameters: np.ndarray,
    loop_birth: float,
    loop_death: float,
    n: int,
    life_pct: float = 0.1,
    n_force_deviate: int = 4,
    k_yen: int = 8,
    loop_lower_pct: float = 5,
    loop_upper_pct: float = 95,
    n_cocycles_used: int = 10,
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
