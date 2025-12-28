import numpy as np

from .frechet import compute_loop_set_frechet


def compute_pairwise_loop_frechet(
    loop_set_a: list[list[list[float]]] | list[np.ndarray],
    loop_set_b: list[list[list[float]]] | list[np.ndarray],
) -> np.ndarray:
    loop_set_a_arr = [
        np.ascontiguousarray(loop, dtype=np.float64) for loop in loop_set_a
    ]
    loop_set_b_arr = [
        np.ascontiguousarray(loop, dtype=np.float64) for loop in loop_set_b
    ]
    pairs = [(a, b) for a in loop_set_a_arr for b in loop_set_b_arr]
    results = compute_loop_set_frechet(pairs)  # type: ignore[call-issue]
    return results
