from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numba import jit
from pygam import LinearGAM, s
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from ..data.analysis_containers import TrajectoryAnalysis


@jit(nopython=True)
def compute_gaussian_weights(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-(distances**2) / (2 * bandwidth**2))


def compute_gene_trends_for_trajectories(
    trajectory_analyses: list[TrajectoryAnalysis],
    coordinates_vertices: np.ndarray,
    gene_expression_matrix: np.ndarray,
    gene_names: list[str],
    values_vertices: np.ndarray,
    confidence_level: float = 0.95,
    bandwidth_scale: float = 1.0,
    verbose: bool = False,
) -> None:
    for traj in trajectory_analyses:
        if verbose:
            logger.info(
                f"Computing gene trends for trajectory with {len(traj.trajectory_coordinates)} points"
            )

        bw = traj.bandwidth_vertices
        if bw is not None:
            bw *= bandwidth_scale

        weights, pseudotime, distances = compute_vertices_weights_for_trajectory(
            trajectory_coords=traj.trajectory_coordinates,
            coordinates_vertices=coordinates_vertices,
            values_vertices=values_vertices,
            bandwidth=bw,
        )

        traj.weights_vertices = weights
        traj.values_vertices = pseudotime
        traj.distances_vertices = distances

        n_genes = len(gene_names)
        n_eval_points = traj.n_bins

        mean_expr = np.zeros((n_genes, n_eval_points))
        se_expr = np.zeros((n_genes, n_eval_points))
        ci_lower = np.zeros((n_genes, n_eval_points))
        ci_upper = np.zeros((n_genes, n_eval_points))

        for gene_idx, gene_name in enumerate(gene_names):
            gene_expr = gene_expression_matrix[:, gene_idx]

            try:
                eval_pts, mean, se, ci_lo, ci_hi = fit_single_gene_gam(
                    pseudotime=pseudotime,
                    expression=gene_expr,
                    weights=weights,
                    n_eval_points=n_eval_points,
                    n_splines=traj.gam_n_splines,
                    confidence_level=confidence_level,
                )

                mean_expr[gene_idx] = mean
                se_expr[gene_idx] = se
                ci_lower[gene_idx] = ci_lo
                ci_upper[gene_idx] = ci_hi
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to fit GAM for gene {gene_name}: {e}")
                continue

        traj.gene_names = gene_names
        traj.mean_expression = mean_expr
        traj.se_expression = se_expr
        traj.ci_lower = ci_lower
        traj.ci_upper = ci_upper


def compute_vertices_weights_for_trajectory(
    trajectory_coords: np.ndarray,
    coordinates_vertices: np.ndarray,
    values_vertices: np.ndarray,
    bandwidth: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = cdist(coordinates_vertices, trajectory_coords)
    min_distances = distances.min(axis=1)

    if bandwidth is None:
        bandwidth = min_distances.std()

    weights = compute_gaussian_weights(min_distances, bandwidth)

    return weights, values_vertices, min_distances


def fit_single_gene_gam(
    pseudotime: np.ndarray,
    expression: np.ndarray,
    weights: np.ndarray,
    n_eval_points: int = 50,
    n_splines: int = 10,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gam = LinearGAM(s(0, n_splines=n_splines, lam=0.6))  # type: ignore[arg-type]
    gam.fit(pseudotime.reshape(-1, 1), expression, weights=weights)

    eval_points = np.linspace(pseudotime.min(), pseudotime.max(), n_eval_points)
    eval_X = eval_points.reshape(-1, 1)

    mean_pred = gam.predict(eval_X)

    ci = gam.confidence_intervals(eval_X, width=confidence_level)
    ci_lower = ci[:, 0]
    ci_upper = ci[:, 1]

    se = (ci_upper - ci_lower) / (2 * 1.96)

    return eval_points, mean_pred, se, ci_lower, ci_upper
