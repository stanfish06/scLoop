# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Mchigan)
import numpy as np
from numba import jit

from .types import Index_t, Size_t


@jit(nopython=True)
def edge_idx_encode(i: Index_t, j: Index_t, num_vertices: Size_t) -> Index_t:
    if i > j:
        i, j = j, i
    return i * num_vertices + j


@jit(nopython=True)
def edge_idx_decode(i: Index_t, num_vertices: Size_t) -> tuple[Index_t, Index_t]:
    return i // num_vertices, i % num_vertices


@jit(nopython=True)
def triangle_idx_encode(
    i: Index_t, j: Index_t, k: Index_t, num_vertices: Size_t
) -> Index_t:
    i, j, k = sorted([i, j, k])
    return i * num_vertices**2 + j * num_vertices + k


@jit(nopython=True)
def triangle_idx_decode(
    i: Index_t, num_vertices: Size_t
) -> tuple[Index_t, Index_t, Index_t]:
    return i // num_vertices**2, (i // num_vertices) % num_vertices, i % num_vertices


@jit(nopython=True)
def decode_edges(simplex_ids, num_vertices):
    result = []
    for sid in simplex_ids:
        result.append(edge_idx_decode(sid, num_vertices))
    return result


@jit(nopython=True)
def decode_triangles(simplex_ids, num_vertices):
    result = []
    for sid in simplex_ids:
        result.append(triangle_idx_decode(sid, num_vertices))
    return result


@jit(nopython=True)
def signed_area_2d(coords: np.ndarray) -> float:
    if coords.shape[0] < 3 or coords.shape[1] < 2:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    area = 0.0
    n = len(x)
    for i in range(n - 1):
        area += x[i] * y[i + 1] - x[i + 1] * y[i]
    return 0.5 * area


@jit(nopython=True)
def encode_triangles_and_edges(triangles, num_vertices):
    trig_ids = []
    edge_ids = []
    for trig in triangles:
        i0 = trig[0]
        i1 = trig[1]
        i2 = trig[2]
        ids = [
            edge_idx_encode(i=i0, j=i1, num_vertices=num_vertices),
            edge_idx_encode(i=i0, j=i2, num_vertices=num_vertices),
            edge_idx_encode(i=i1, j=i2, num_vertices=num_vertices),
        ]
        edge_ids.append(ids)
        trig_ids.append(
            triangle_idx_encode(i=i0, j=i1, k=i2, num_vertices=num_vertices)
        )
    return edge_ids, trig_ids


@jit(nopython=True)
def extract_edges_from_coo(rows, cols, data):
    n = len(rows)
    edges = np.empty((n, 2), dtype=np.int64)
    weights = np.empty(n, dtype=np.float64)

    count = 0
    for k in range(n):
        i, j, w = rows[k], cols[k], data[k]
        if i >= j:
            continue
        edges[count, 0] = i
        edges[count, 1] = j
        weights[count] = w
        count += 1

    return edges[:count], weights[:count]


@jit(nopython=True)
def loop_vertices_to_edge_ids(
    loop_vertices: np.ndarray, num_vertices: Size_t
) -> np.ndarray:
    """
    Encode a loop represented by vertex indices into edge ids (edge_idx_encode),
    closing the loop back to the starting vertex.
    """
    n = loop_vertices.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    is_closed = n > 1 and loop_vertices[0] == loop_vertices[n - 1]
    n_edges = n - 1 if is_closed else n

    edge_ids = np.empty(n_edges, dtype=np.int64)
    for k in range(n_edges):
        i = loop_vertices[k]
        j = loop_vertices[(k + 1) % n]
        edge_ids[k] = edge_idx_encode(int(i), int(j), num_vertices)
    return edge_ids


@jit(nopython=True)
def edge_ids_to_rows(edge_ids: np.ndarray, edge_row_ids: np.ndarray) -> np.ndarray:
    """
    Map encoded edge ids to row indices
    where index is edge id and value is row index or -1 if absent.
    """
    n = edge_ids.shape[0]
    rows = np.empty(n, dtype=np.int64)
    count = 0
    for k in range(n):
        row = edge_row_ids[edge_ids[k]]
        if row >= 0:
            rows[count] = row
            count += 1
    return rows[:count]


@jit(nopython=True)
def loops_masks_to_edges_masks(loops_mask: np.ndarray) -> list[np.ndarray]:
    """Convert loop order masks to per-edge binary masks in traversal order.

    loops_mask contains traversal order (1, 2, 3, ...) with 0 for edges not in loop.
    """
    n_loops, n_edges = loops_mask.shape
    result = []
    for i in range(n_loops):
        row = loops_mask[i, :]
        edge_indices = np.where(row > 0)[0]
        n_true_edges = len(edge_indices)

        # Sort by traversal order
        orders = row[edge_indices]
        sort_indices = np.argsort(orders)
        edge_indices_sorted = edge_indices[sort_indices]

        edge_masks = np.zeros((n_true_edges, n_edges), dtype=np.bool_)
        for j in range(n_true_edges):
            edge_masks[j, edge_indices_sorted[j]] = True
        result.append(edge_masks)
    return result


@jit(nopython=True)
def nearest_neighbor_per_row(
    distance_matrix: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = distance_matrix.shape
    neighbor_indices = np.empty((n_rows, k), dtype=np.int64)
    neighbor_distances = np.empty((n_rows, k), dtype=np.float64)

    for si in range(n_rows):
        distances = distance_matrix[si, :]
        valid_count = 0
        valid_indices = np.empty(n_cols, dtype=np.int64)
        valid_distances = np.empty(n_cols, dtype=np.float64)

        for j in range(n_cols):
            if not np.isnan(distances[j]):
                valid_indices[valid_count] = j
                valid_distances[valid_count] = distances[j]
                valid_count += 1

        if valid_count == 0:
            neighbor_indices[si, :] = -1
            neighbor_distances[si, :] = np.nan
            continue

        valid_indices = valid_indices[:valid_count]
        valid_distances = valid_distances[:valid_count]

        n_keep = min(valid_count, k)
        sorted_idx = np.argsort(valid_distances)[:n_keep]

        for idx in range(n_keep):
            neighbor_indices[si, idx] = valid_indices[sorted_idx[idx]]
            neighbor_distances[si, idx] = valid_distances[sorted_idx[idx]]

        for idx in range(n_keep, k):
            neighbor_indices[si, idx] = -1
            neighbor_distances[si, idx] = np.nan

    return neighbor_indices, neighbor_distances


# this cannot be jitted
def loops_to_coords(
    embedding: np.ndarray, loops_vertices: list[list[int]]
) -> list[list[list[float]]]:
    loops_coords = []
    for vertices in loops_vertices:
        coords = np.asarray(embedding[np.asarray(vertices, dtype=int)])
        # properly form a loop if possible - only close if not already closed
        if coords.shape[0] > 2:
            # Check if loop is already closed (first == last vertex)
            if len(vertices) < 2 or vertices[0] != vertices[-1]:
                coords = np.vstack([coords, coords[[0]]])
        loops_coords.append(coords.tolist())
    return loops_coords


@jit(nopython=True)
def loop_vertices_to_edge_ids_with_signs(
    loop_vertices: np.ndarray, num_vertices: Size_t
) -> tuple[np.ndarray, np.ndarray]:
    n = loop_vertices.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)

    is_closed = n > 1 and loop_vertices[0] == loop_vertices[n - 1]
    n_edges = n - 1 if is_closed else n

    edge_ids = np.empty(n_edges, dtype=np.int64)
    edge_signs = np.empty(n_edges, dtype=np.int8)
    for k in range(n_edges):
        i = int(loop_vertices[k])
        j = int(loop_vertices[(k + 1) % n])
        if i < j:
            edge_ids[k] = i * num_vertices + j
            edge_signs[k] = 1
        else:
            edge_ids[k] = j * num_vertices + i
            edge_signs[k] = -1
    return edge_ids, edge_signs


@jit(nopython=True)
def smooth_along_loop_1d(values: np.ndarray, half_window: int) -> np.ndarray:
    n = values.shape[0]
    smoothed = np.zeros(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        count = 0
        for offset in range(-half_window, half_window + 1):
            idx = (i + offset) % n
            total += values[idx]
            count += 1
        smoothed[i] = total / count
    return smoothed


@jit(nopython=True)
def smooth_along_loop_2d(values: np.ndarray, half_window: int) -> np.ndarray:
    n, d = values.shape
    smoothed = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        for offset in range(-half_window, half_window + 1):
            idx = (i + offset) % n
            for j in range(d):
                smoothed[i, j] += values[idx, j]
        for j in range(d):
            smoothed[i, j] /= 2 * half_window + 1
    return smoothed
