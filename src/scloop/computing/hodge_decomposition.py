# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import multiprocessing
import queue

import numpy as np
from loguru import logger
from numba import jit
from scipy.sparse import csr_matrix

from ..data.constants import (
    DEFAULT_MAXITER_EIGENDECOMPOSITION,
    DEFAULT_N_HODGE_COMPONENTS,
    DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    NUMERIC_EPSILON,
)
from ..data.types import Diameter_t


def run_eigsh_worker(
    q: multiprocessing.Queue,
    hodge_matrix: csr_matrix,
    k: int,
    tol: float,
    maxiter: int | None,
) -> None:
    try:
        from scipy.sparse.linalg import eigsh

        vals, vecs = eigsh(hodge_matrix, k=k, which="SA", tol=tol, maxiter=maxiter)
        q.put(("success", (vals, vecs)))
    except Exception as e:
        q.put(("error", e))


def compute_hodge_matrix(
    boundary_matrix_d0: csr_matrix,
    boundary_matrix_d1: csr_matrix,
    triangle_diams: np.ndarray,
    thresh: Diameter_t,
    normalized: bool = True,
) -> csr_matrix:
    bd1 = boundary_matrix_d0
    bd2_full = boundary_matrix_d1

    bd1_bd2 = bd1.dot(bd2_full)
    if bd1_bd2.count_nonzero() != 0:
        raise ValueError(
            f"d1 @ d2 has {bd1_bd2.count_nonzero()} nonzero entries. "
            f"Simplex orientation is incorrect."
        )

    cols_use = np.where(triangle_diams <= thresh)[0]

    if cols_use.size == 0:
        logger.warning(
            f"No triangles below threshold {thresh}. hodge_matrix_d1 is d1T*d1 only."
        )
        bd2 = csr_matrix(bd2_full.shape)
    else:
        bd2 = bd2_full[:, cols_use]

    if normalized:
        D2 = np.maximum(abs(bd2).sum(1), 1)
        D1 = 2 * (abs(bd1) @ D2)
        D3 = 1 / 3
        L1 = (bd1.T.multiply(D2).multiply(1 / D1.T)) @ bd1 + (
            (bd2 * D3) @ bd2.T
        ).multiply(1 / D2.T)
        L1 = L1.multiply(1 / np.sqrt(D2)).multiply(np.sqrt(D2).T)
        hodge_matrix_d1: csr_matrix = csr_matrix(L1)
    else:
        hodge_matrix_d1 = csr_matrix(bd1.transpose() @ bd1 + bd2 @ bd2.transpose())

    return hodge_matrix_d1


def compute_hodge_eigendecomposition(
    hodge_matrix: csr_matrix,
    n_components: int = DEFAULT_N_HODGE_COMPONENTS,
    timeout: float = DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    maxiter: int | None = DEFAULT_MAXITER_EIGENDECOMPOSITION,
) -> tuple[np.ndarray, np.ndarray] | None:
    if hodge_matrix.shape[0] < 2:
        logger.warning("hodge_matrix too small for eigendecomposition (shape < 2).")
        return None

    k = min(n_components, hodge_matrix.shape[0] - 2)
    if k <= 0:
        logger.warning(f"Not enough dimensions for eigendecomposition (k={k}).")
        return None

    tolerances = [1e-6, 1e-5, 1e-4, 1e-3]

    for tol in tolerances:
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_eigsh_worker,
            args=(q, hodge_matrix, k, tol, maxiter),
        )
        p.start()

        try:
            status, result = q.get(timeout=timeout)
            p.join()
            if status == "success":
                eigenvalues, eigenvectors = result
                sort_idx = np.argsort(eigenvalues)
                return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
            else:
                logger.warning(
                    f"Eigendecomposition failed with tol={tol}: {result}. Retrying..."
                )
                continue
        except queue.Empty:
            logger.warning(
                f"Eigendecomposition timed out ({timeout}s) with tol={tol}. "
                "Retrying with looser tolerance."
            )
            p.terminate()
            p.join()
            continue
        except Exception as e:
            logger.warning(
                f"Eigendecomposition failed with tol={tol}: {e}. Retrying..."
            )
            if p.is_alive():
                p.terminate()
                p.join()
            continue

    logger.error("Eigendecomposition failed after all retries.")
    return None


@jit(nopython=True)
def compute_weighted_hodge_embedding(
    edge_evecs: np.ndarray,
    eigenvalues: np.ndarray,
    edge_gradients: np.ndarray,
    epsilon: float = NUMERIC_EPSILON,
    power_evals: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes weighted edge embedding using hodge eigenvectors and edge gradients

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (signed_embedding, involvement_embedding)
    """
    n_edges = edge_evecs.shape[0]
    if n_edges < 2:
        return np.zeros(n_edges), np.zeros(n_edges)

    _combined = np.concatenate((np.abs(edge_gradients), np.abs(edge_evecs)), axis=1)

    _cors = np.corrcoef(_combined, rowvar=False)[0, 1:]

    _weights_evals = 1.0 / (np.power(eigenvalues, power_evals) + epsilon)

    weights = _weights_evals * _cors
    weights = weights / (np.sum(np.abs(weights)) + epsilon)

    signed_embedding = np.sum(edge_evecs * weights, axis=1)

    involvement_embedding = np.sum(np.abs(edge_evecs) * np.abs(weights), axis=1)

    return signed_embedding, involvement_embedding
