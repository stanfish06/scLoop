# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import numpy as np
from numba import jit

from ..data.constants import NUMERIC_EPSILON


@jit(nopython=True)
def compute_weighted_hodge_embedding(
    edge_evecs: np.ndarray,
    eigenvalues: np.ndarray,
    edge_gradients: np.ndarray,
    epsilon: float = NUMERIC_EPSILON,
    power_evals: float = 1,
) -> np.ndarray:
    """Computes weighted edge embedding using hodge eigenvectors and edge gradients

    Parameters
    ----------
    edge_evacs : numpy array of shape (n_edges, n_components)
        Description of parameter.
    eigenvalues : numpy array of shape (n_components, 1)
        Description of parameter.
    edge_gradients : numpy array of shape (n_edges,)
        Description of parameter.
    Returns
    -------
    numpy array of shape (n_edges, 1)
        Description of return value.
    """
    _combined = np.concatenate((edge_gradients, edge_evecs), axis=1)
    _cors = np.corrcoef(_combined, rowvar=False)[0, 1:]
    _weights_evals = 1.0 / (np.power(eigenvalues, power_evals) + epsilon)
    weights = _weights_evals * _cors
    weights = weights / np.sum(np.abs(weights))
    return np.sum(edge_evecs * weights, axis=1)
