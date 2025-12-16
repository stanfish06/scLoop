# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
cimport cython
from libc.stdlib cimport malloc, free

cdef extern from "m4ri/m4ri.h":
    ctypedef int rci_t
    ctypedef int wi_t
    ctypedef unsigned long long m4ri_word "word"
    ctypedef int BIT
    ctypedef struct mzd_t:
        rci_t nrows
        rci_t ncols
        wi_t width

    mzd_t *mzd_init(rci_t, rci_t) nogil
    void mzd_free(mzd_t *) nogil
    mzd_t *mzd_copy(mzd_t *DST, const mzd_t *A) nogil
    void mzd_write_bit(mzd_t *m, rci_t row, rci_t col, BIT value) nogil
    BIT mzd_read_bit(mzd_t *M, rci_t row, rci_t col) nogil

    void mzd_print(mzd_t *) nogil
    int mzd_solve_left(mzd_t *A, mzd_t *B, int cutoff, int inconsistency_check) nogil

cdef int solve_gf2_nogil(
    const rci_t[:] one_ridx_A,
    const rci_t[:] one_cidx_A,
    rci_t nrow_A,
    rci_t ncol_A,
    const rci_t[:] one_idx_b,
    BIT* solution_out
) noexcept nogil:
    cdef mzd_t *A = mzd_init(nrow_A, ncol_A)
    cdef mzd_t *b = mzd_init(nrow_A, 1)
    cdef size_t i, nnz_A = one_ridx_A.shape[0], nnz_b = one_idx_b.shape[0]
    cdef int result

    for i in range(nnz_A):
        mzd_write_bit(A, one_ridx_A[i], one_cidx_A[i], 1)

    for i in range(nnz_b):
        mzd_write_bit(b, one_idx_b[i], 0, 1)

    result = mzd_solve_left(A, b, 0, 1)

    if result == 0:
        for i in range(<size_t>ncol_A):
            solution_out[i] = mzd_read_bit(b, i, 0)

    mzd_free(A)
    mzd_free(b)
    return result


cdef int solve_gf2_nogil_single_b(
    mzd_t* A_base,
    rci_t nrow_A,
    rci_t ncol_A,
    const rci_t[:] one_idx_b,
    BIT* solution_out
) noexcept nogil:
    cdef mzd_t *A = mzd_init(nrow_A, ncol_A)
    cdef mzd_t *b = mzd_init(nrow_A, 1)
    cdef size_t i, nnz_b = one_idx_b.shape[0]
    cdef int result

    mzd_copy(A, A_base)

    for i in range(nnz_b):
        mzd_write_bit(b, one_idx_b[i], 0, 1)

    result = mzd_solve_left(A, b, 0, 1)

    if result == 0:
        for i in range(<size_t>ncol_A):
            solution_out[i] = mzd_read_bit(b, i, 0)

    mzd_free(A)
    mzd_free(b)
    return result


def solve_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b):
    assert nrow_A >= ncol_A, "number of rows must be greater than or equal to the number of columns"

    cdef rci_t[:] ridx_view
    cdef rci_t[:] cidx_view
    cdef rci_t[:] b_idx_view
    cdef BIT* solution
    cdef int result
    cdef list sol_list
    cdef mzd_t *A
    cdef mzd_t *b
    cdef rci_t nrow_c
    cdef rci_t ncol_c

    try:
        import numpy as np
        ridx_view = np.asarray(one_ridx_A, dtype=np.int32)
        cidx_view = np.asarray(one_cidx_A, dtype=np.int32)
        b_idx_view = np.asarray(one_idx_b, dtype=np.int32)
        nrow_c = nrow_A
        ncol_c = ncol_A
        solution = <BIT*>malloc(ncol_A * sizeof(BIT))

        if solution == NULL:
            raise MemoryError("Failed to allocate solution array")
        try:
            with nogil:
                result = solve_gf2_nogil(ridx_view, cidx_view, nrow_c, ncol_c, b_idx_view, solution)
            if result == 0:
                sol_list = [solution[i] for i in range(ncol_A)]
            else:
                sol_list = None
            return (result == 0, sol_list)
        finally:
            free(solution)
    except:
        A = mzd_init(nrow_A, ncol_A)
        b = mzd_init(nrow_A, 1)
        for (i, j) in zip(one_ridx_A, one_cidx_A):
            mzd_write_bit(A, i, j, 1)
        for i in one_idx_b:
            mzd_write_bit(b, i, 0, 1)
        try:
            result = mzd_solve_left(A, b, 0, 1)
            if result == 0:
                sol_list = [mzd_read_bit(b, i, 0) for i in range(ncol_A)]
            else:
                sol_list = None
            return (result == 0, sol_list)
        finally:
            mzd_free(A)
            mzd_free(b)


def solve_multiple_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b_list):
    cdef rci_t[:] ridx_view
    cdef rci_t[:] cidx_view
    cdef mzd_t *A_base
    cdef mzd_t *A
    cdef mzd_t *b
    cdef BIT* solution
    cdef int result
    cdef size_t i, j, n_systems = len(one_idx_b_list)
    cdef rci_t[:] b_idx_view
    cdef rci_t nrow_c
    cdef rci_t ncol_c

    results = []
    sols = []

    try:
        import numpy as np
        ridx_view = np.asarray(one_ridx_A, dtype=np.int32)
        cidx_view = np.asarray(one_cidx_A, dtype=np.int32)
        nrow_c = nrow_A
        ncol_c = ncol_A
        solution = <BIT*>malloc(ncol_A * sizeof(BIT))
        if solution == NULL:
            raise MemoryError("Failed to allocate solution array")
        try:
            A_base = mzd_init(nrow_A, ncol_A)
            for i in range(ridx_view.shape[0]):
                mzd_write_bit(A_base, ridx_view[i], cidx_view[i], 1)
            for one_idx_b in one_idx_b_list:
                b_idx_view = np.asarray(one_idx_b, dtype=np.int32)
                with nogil:
                    result = solve_gf2_nogil_single_b(A_base, nrow_c, ncol_c, b_idx_view, solution)
                if result == 0:
                    sol = [solution[j] for j in range(ncol_A)]
                else:
                    sol = None
                results.append(result)
                sols.append(sol)
            mzd_free(A_base)
            return results, sols
        finally:
            free(solution)
    except:
        A_base = mzd_init(nrow_A, ncol_A)
        for (i, j) in zip(one_ridx_A, one_cidx_A):
            mzd_write_bit(A_base, i, j, 1)
        try:
            for one_idx_b in one_idx_b_list:
                A = mzd_init(nrow_A, ncol_A)
                b = mzd_init(nrow_A, 1)
                try:
                    mzd_copy(A, A_base)
                    for i in one_idx_b:
                        mzd_write_bit(b, i, 0, 1)
                    result = mzd_solve_left(A, b, 0, 1)
                    if result == 0:
                        sol = [mzd_read_bit(b, i, 0) for i in range(ncol_A)]
                    else:
                        sol = None
                    results.append(result)
                    sols.append(sol)
                finally:
                    mzd_free(A)
                    mzd_free(b)
        finally:
            mzd_free(A_base)
        return results, sols
