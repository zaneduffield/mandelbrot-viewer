import numpy as np
from numba import njit, prange

from .perturbation_state import PerturbationState
from brot.utils.constants import (
    GLITCH_ITER,
    BREAKOUT_R2,
)


@njit
def get_init_delta(data: PerturbationState, i, j):
    hor_delta = (i - data.ref_coords[1]) * data.w_per_pix
    ver_delta = (data.ref_coords[0] - j) * data.h_per_pix
    return complex(hor_delta, ver_delta)


@njit
def get_delta_estimate(data: PerturbationState, init_delta, iteration):
    scaled_delta = init_delta * data.scaling_factor
    out = 0
    for k in range(data.num_terms):
        # TODO this is the wrong order to iterate through the terms array
        term = data.terms[k][iteration - 1]
        out += term * scaled_delta
        scaled_delta *= init_delta * data.scaling_factor
    return out


@njit
def get_reference(data: PerturbationState, iteration):
    return data.precise_reference[iteration - 1]


@njit
def sq_mod(point):
    return point.real * point.real + point.imag * point.imag


@njit
def approximate_pixel(data: PerturbationState, x, y, iterations_grid, points):
    delta_0 = get_init_delta(data, x, y)
    delta_i = get_delta_estimate(data, delta_0, data.iter_accurate)
    point = 0
    i = data.iter_accurate
    while i <= data.breakout:
        ref_i = get_reference(data, i)
        point = delta_i + ref_i
        actual_size = sq_mod(point)

        if actual_size > BREAKOUT_R2:
            break

        if actual_size < data.glitch_error_threshold * sq_mod(ref_i) or (
            i == data.breakout and data.breakout < data.max_iterations
        ):
            iterations_grid[y, x] = GLITCH_ITER
            return

        delta_i = 2 * get_reference(data, i) * delta_i + delta_i * delta_i + delta_0
        i += 1

    if i == data.iter_accurate and i <= data.breakout:
        # broke out before iterating, find true breakout value using binary search on accurate estimations
        lo, hi = 1, i - 1
        while hi != lo:
            mid = (lo + hi) // 2
            point = get_reference(data, mid) + get_delta_estimate(data, delta_0, mid)
            if sq_mod(point) <= BREAKOUT_R2:
                lo = mid + 1
            else:
                hi = mid

        point = get_reference(data, hi) + get_delta_estimate(data, delta_0, hi)
        i = hi

    iterations_grid[y, x] = i - 1
    points[y, x] = point


@njit(parallel=True)
def iterate_perturbation(
    iterations_grid: np.ndarray,
    points: np.ndarray,
    state: PerturbationState,
    fix_glitches: bool,
):
    for y in prange(state.height):
        for x in range(state.width):
            if fix_glitches and iterations_grid[y, x] != GLITCH_ITER:
                continue
            approximate_pixel(state, x, y, iterations_grid, points)

    return iterations_grid, points
