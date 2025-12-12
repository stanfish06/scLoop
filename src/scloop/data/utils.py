# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Mchigan)
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
