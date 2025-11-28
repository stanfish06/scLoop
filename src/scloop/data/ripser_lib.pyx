# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Mchigan)
cimport cython
from libcpp.vector cimport vector 
from scipy.sparse import coo_matrix
import numpy as np
import typing
import dataclasses

ctypedef float value_t
cdef extern from "ripser.hpp":
    cdef cppclass ripserResults:
        vector[vector[value_t]] births_and_deaths_by_dim
        vector[vector[vector[int]]] cocycles_by_dim
        int num_edges
    cdef ripserResults rips_dm_sparse(int* I, int* J, float* V, int NEdges, int N, int modulus, int dim_max, float threshold, int do_cocycles)

@dataclasses.dataclass
class RipserResults:
    births_and_deaths_by_dim: list
    cocycles_by_dim: list
    num_edges: int

cdef list converting_cocycles_to_list(vector[vector[vector[int]]] cocycles_by_dim, int dim): 
    '''
    This is a vector of representative cocycles for each
    dimension. For now, only cocycles above dimension 0 are added, so
    dimension 0 is an empty list For the others, cocycles_by_dim[d] holds an
    array of representative cocycles for dimension d which is parallel with
    the array of births/deaths for dimension d. Each element of the array is
    itself an array of unrolled information about the cocycle For dimension 1,
    for example, the zeroeth element of the array contains [ccl0_simplex0_idx0
    ccl0_simplex0_idx1 ccl0_simplex0_val, ccl0_simplex1_idx0
    ccl0_simplex1_idx1 ccl0_simplex1_val, ... ccl0_simplexk_idx0
    ccl0_simplexk_idx1 ccl0_simplexk_val] for a cocycle representing the first
    persistence point, which has k simplices with nonzero values in the
    representative cocycle
    '''
    cdef list cocycle_representatives = []
    cdef vector[vector[int]]* cocycles = &cocycles_by_dim[dim]
    cdef int nc = cocycles.size() 
    cdef vector[int]* rep_i
    cdef int chunk_size
    cdef int n_simplices
    cdef list cocycle_rep_members
    cdef list simplex
    cdef int start_idx
    cdef int end_idx

    for i in range(nc):
        rep_i = &cocycles[0][i]
        chunk_size = dim + 2
        n_simplices = rep_i.size() // chunk_size
        cocycle_rep_members = []
        for j in range(n_simplices):
            simplex = []
            start_idx = j * chunk_size
            end_idx = start_idx + chunk_size - 1
            for k in range(start_idx, end_idx):
                simplex.append(int(rep_i[0][k]))
            cocycle_rep_members.append([simplex, int(rep_i[0][end_idx])])
        cocycle_representatives.append(cocycle_rep_members)
    return cocycle_representatives

cdef list converting_birth_death_to_list(vector[vector[value_t]] births_and_deaths_by_dim, int dim):
    '''
    This is a vector of unrolled persistence diagrams
    so, for example births_and_deaths_by_dim[0] contains a list of
    [birth0, death0, birth1, death1, ..., birthk, deathk]
    for k points in the 0D persistence diagram
    and likewise for d-dimensional persistence in births_and_deaths_by_dim[d]
    '''
    cdef list birth = []
    cdef list death = []
    cdef vector[value_t]* birth_death = &births_and_deaths_by_dim[dim]
    cdef int n_pairs = birth_death.size() // 2
    for i in range(n_pairs):
        birth.append(birth_death[0][i * 2])
        death.append(birth_death[0][i * 2 + 1])
    return [birth, death]

def ripser(
    distance_matrix: coo_matrix,
    int modulus,
    int dim_max,
    float threshold,
    bool do_cocycles,
) -> RipserResults:
    # I, J, and V need to be contiguous array
    cdef int[::1] _I = np.ascontiguousarray(distance_matrix.row, dtype = np.intc)
    cdef int[::1] _J = np.ascontiguousarray(distance_matrix.col, dtype = np.intc)
    cdef float[::1] _V = np.ascontiguousarray(distance_matrix.data, dtype = np.float32)

    cdef int* I = &_I[0]
    cdef int* J = &_J[0]
    cdef float* V = &_V[0]
    
    cdef int NEdges = distance_matrix.nnz
    cdef int N = distance_matrix.shape[0]
    cdef ripserResults res = rips_dm_sparse(I, J, V, NEdges, N, modulus, dim_max, threshold, int(do_cocycles))
    cdef list persistence_diagrams = [converting_birth_death_to_list(res.births_and_deaths_by_dim, i) for i in range(dim_max + 1)]
    cdef list cocycle_representatives = [converting_cocycles_to_list(res.cocycles_by_dim, i) for i in range(dim_max + 1)]
    return RipserResults(
        persistence_diagrams,
        cocycle_representatives,
        NEdges
    )
