import math
import time
from pathlib import Path

import numpy as np
from gmpy2 import mpc, mpfr
from numba import njit, prange, float64, complex128, int32, jit
from numba.experimental import jitclass
from dataclasses import dataclass

from opencl.mandelbrot_cl import MandelbrotCL, cl
from utils.constants import (
    GLITCH_ITER,
    MAX_GLITCH_FIX_LOOPS,
    MAX_OK_GLITCH_COUNT,
    BREAKOUT_R2,
    GLITCH_DIFF_THRESHOLD,
)
from utils.mandelbrot_utils import MandelbrotConfig, get_precision, my_logger


@dataclass(eq=False)
class Reference:
    orbit: np.ndarray
    series_constants: np.ndarray
    accurate_iters: int
    escaped_at: int
    init: mpc
    width: mpfr
    scaling_factor: float
    precision: int

    def __hash__(self):
        return hash(
            (str(self.init), self.width, self.scaling_factor, self.accurate_iters)
        )


@jit(forceobj=True)
def iterate_ref(init_ref: mpc, iterations):
    ref_hist = np.zeros(iterations, dtype=np.complex128)
    ref = init_ref
    for i in range(iterations):
        temp = complex(ref)
        ref_hist[i] = temp
        if temp.real * temp.real + temp.imag * temp.imag > BREAKOUT_R2:
            return ref_hist, i + 1

        ref = ref * ref + init_ref

    return ref_hist, iterations


@njit
def iterate_series_constants(
    ref_hist,
    ref_escaped_at: int,
    probe_deltas,
    terms,
    num_terms: int,
    scaling_factor: float,
):
    """
    The error tolerance has a huge impact on the correctness of the series approximation
    Pretty much no matter how hard we try this won't be good enough in all cases and glitches
    will slip through. There are more advanced techniques for detecting when the series approximation is no longer
    accurate but I don't understand them yet.
    """
    tolerance = 1 / np.power(2, 48)

    scaled_delta_powers = np.zeros((len(probe_deltas), num_terms), dtype=np.complex128)
    for j in range(len(probe_deltas)):
        scaled_delta_powers[j, 0] = probe_deltas[j] * scaling_factor
        for k in range(1, num_terms):
            scaled_delta_powers[j, k] = (
                scaled_delta_powers[j, k - 1] * scaled_delta_powers[j, 0]
            )

    # This acts to reduce the size of our series terms and is reversed when computing using them
    terms[0][0] = 1 / scaling_factor
    for i in range(1, num_terms):
        terms[i][0] = 0

    for i in range(ref_escaped_at):
        z_comp = ref_hist[i]
        terms[0][i + 1] = 2 * z_comp * terms[0][i] + terms[0][0]
        for j in range(1, num_terms):
            s = 0
            for k in range(j):
                s += terms[k][i] * terms[j - k - 1][i]

            new_term = 2 * z_comp * terms[j][i] + s
            if not np.isnan(new_term):
                terms[j][i + 1] = new_term

        for j in range(len(scaled_delta_powers)):
            last = 1 / tolerance
            for k in range(num_terms):
                term = np.abs(terms[k][i + 1] * scaled_delta_powers[j][k])
                if last * tolerance < term:
                    return i
                last = term

    return ref_escaped_at


def get_probe_deltas(t_left: mpc, b_right: mpc, ref_init: mpc, num_probes):
    probes = []
    square_side_len = int(math.sqrt(num_probes))
    for i in range(square_side_len):
        for j in range(square_side_len):
            x_ratio, y_ratio = i / square_side_len, j / square_side_len
            probe_del = (
                x_ratio * (b_right.real - t_left.real)
                - y_ratio * (t_left.imag - b_right.imag) * 1j
            )
            probes.append(complex(t_left + probe_del - ref_init))

    return np.array(probes)


def compute_series_constants(
    t_left: mpc,
    b_right: mpc,
    ref_init: mpc,
    ref_hist: np.ndarray,
    ref_escaped_at: int,
    iterations: int,
    num_terms,
    num_probes,
):
    p_deltas_init = get_probe_deltas(t_left, b_right, ref_init, num_probes)
    terms = np.zeros((num_terms, iterations + 1), dtype=np.complex_, order="F")

    scaling_factor = float(1 / (b_right.real - t_left.real))
    accurate_iters = iterate_series_constants(
        ref_hist, ref_escaped_at, p_deltas_init, terms, num_terms, scaling_factor
    )
    return Reference(
        orbit=ref_hist,
        series_constants=terms,
        accurate_iters=accurate_iters,
        init=ref_init,
        escaped_at=ref_escaped_at,
        width=(b_right.real - t_left.real),
        scaling_factor=scaling_factor,
        precision=get_precision(),
    )


class PerturbationComputer:
    def __init__(self):
        self._cl = None

    def _get_cl(self):
        if self._cl is None:
            self._cl = PerturbationCL()
        return self._cl

    def _get_or_set_zero_arr(self, name, dtype: type, shape):
        if not hasattr(self, name) or self.__getattribute__(name).shape != shape:
            self.__setattr__(name, np.empty(shape, dtype=dtype))
        self.__getattribute__(name).fill(0)
        return self.__getattribute__(name)

    def _get_iterations_grid(self, config: MandelbrotConfig):
        shape = (config.image_height, config.image_width)
        return self._get_or_set_zero_arr("_iterations_grid", np.int32, shape)

    def _get_points_grid(self, config: MandelbrotConfig):
        shape = (config.image_height, config.image_width)
        return self._get_or_set_zero_arr("_points_grid", np.complex128, shape)

    def _get_ref(self, config: MandelbrotConfig, coords, num_series_terms, num_probes):
        config.set_precision()
        ref = config.get_point_by_coords(*coords)
        start = time.time()
        my_logger.debug("iterating reference")
        ref_hist, ref_escaped_at = iterate_ref(ref, config.max_iterations)
        my_logger.debug(f"iterating reference took {time.time() - start} seconds")
        my_logger.debug("computing series constants...")
        start = time.time()
        reference = compute_series_constants(
            config.t_left(),
            config.b_right(),
            ref,
            ref_hist,
            ref_escaped_at,
            config.max_iterations,
            num_series_terms,
            num_probes,
        )
        my_logger.debug(f"series constants took {time.time() - start} seconds")

        my_logger.debug(
            f"proceeding with {reference.accurate_iters} reference iterations"
        )
        my_logger.debug(f"reference broke out at {ref_escaped_at}")
        return reference

    def compute(
        self,
        config: MandelbrotConfig,
        num_probes,
        num_series_terms,
    ):
        ref_coords = config.image_height // 2, config.image_width // 2
        loops = 0
        while loops <= MAX_GLITCH_FIX_LOOPS:
            reference = self._get_ref(config, ref_coords, num_series_terms, num_probes)
            pertubation_state = get_perturbation_state(
                config, np.array(ref_coords, dtype=np.int32), reference
            )
            my_logger.debug(f"new ref iterated at: {pertubation_state.ref_coords}")

            if config.gpu:
                iterations_grid, points = self._get_cl().compute(
                    pertubation_state,
                    fix_glitches=bool(loops),
                    double_precision=config.gpu_double_precision,
                )
            else:
                iterations_grid, points = (
                    self._get_iterations_grid(config),
                    self._get_points_grid(config),
                )
                approximate_pixels(
                    iterations_grid,
                    points,
                    pertubation_state,
                    fix_glitches=bool(loops),
                )
            yield iterations_grid, points

            glitched_count = np.sum(iterations_grid == GLITCH_ITER)
            my_logger.debug(f"{glitched_count} pixels remaining")
            if glitched_count <= MAX_OK_GLITCH_COUNT:
                break

            ref_coords = get_new_ref(iterations_grid)
            my_logger.debug(f"new ref at :{ref_coords}")
            loops += 1


def get_new_ref(iterations_grid: np.ndarray):
    lo = np.array([0, 0], dtype=np.int64)
    hi = np.array(iterations_grid.shape, dtype=np.int64)
    start = time.time()
    refs = _get_new_refs(iterations_grid == GLITCH_ITER, lo, hi)
    my_logger.debug(f"found {len(refs)} new refs in {time.time() - start} seconds")
    return max(refs)[1:]


@njit()
def _get_search_size(lo, hi):
    return (hi[0] - lo[0]) * (hi[1] - lo[1])


@njit(inline="always")
def _add_new_ref(lo, hi, refs):
    size = _get_search_size(lo, hi)
    refs.append((size, (hi[0] + lo[0]) // 2, (hi[1] + lo[1]) // 2))


@njit()
def _add_new_search(lo, hi, searches, refs):
    if len(refs) == 0 or _get_search_size(lo, hi) > max(refs)[0]:
        searches.append((lo, hi))


@njit(parallel=True)
def _get_new_refs(
    blob_grid: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_sectors=20, max_refs=5
):
    searches = [(lo, hi)]
    refs = []
    while searches and len(refs) < max_refs:
        (lo, hi) = searches.pop()
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            _add_new_ref(lo, hi, refs)
            continue

        dim = int(hi[0] - lo[0] < hi[1] - lo[1])
        num_sectors = min(hi[dim] - lo[dim], max_sectors)
        blob_counts = np.zeros(num_sectors)
        sector_width = (hi[dim] - lo[dim]) / num_sectors
        sector = 0
        for i in prange(lo[0], hi[0]):
            if not dim:
                sector = int((i - lo[0]) / sector_width)
            for j in range(lo[1], hi[1]):
                if dim:
                    sector = int((j - lo[1]) / sector_width)
                blob_counts[sector] += int(blob_grid[i, j])

        total = np.sum(blob_counts)
        if total == 0 or total / _get_search_size(lo, hi) >= 0.9:
            _add_new_ref(lo, hi, refs)
            continue

        selection = None
        index = 0
        while index <= num_sectors:
            if index < num_sectors and blob_counts[index] / total > 1 / num_sectors:
                if selection is None:
                    selection = [index, index + 1]
                else:
                    selection[-1] = index + 1
            elif selection is not None:
                new_lo = lo.copy()
                new_hi = hi.copy()
                new_lo[dim] += selection[0] * sector_width
                new_hi[dim] = lo[dim] + selection[1] * sector_width
                _add_new_search(new_lo, new_hi, searches, refs)
                selection = None

            index += 1

    return refs


def get_perturbation_state(config: MandelbrotConfig, ref_coords, reference: Reference):
    return PerturbationState(
        config.get_width_per_pix(),
        config.get_height_per_pix(),
        config.image_width,
        config.image_height,
        ref_coords,
        reference.series_constants,
        reference.series_constants.shape[0],
        reference.escaped_at,
        reference.orbit,
        reference.accurate_iters,
        config.max_iterations,
        reference.scaling_factor,
    )


@jitclass(
    [
        ("w_per_pix", float64),
        ("h_per_pix", float64),
        ("width", int32),
        ("height", int32),
        ("ref_coords", int32[:]),
        ("terms", complex128[:, :]),
        ("num_terms", int32),
        ("breakout", int32),
        ("precise_reference", complex128[:]),
        ("iter_accurate", int32),
        ("max_iterations", int32),
        ("scaling_factor", float64),
    ]
)
class PerturbationState:
    def __init__(
        self,
        w_per_pix,
        h_per_pix,
        width,
        height,
        ref_coords,
        terms,
        num_terms,
        breakout,
        precise_reference,
        iter_accurate,
        max_iterations,
        scaling_factor,
    ):
        self.w_per_pix = w_per_pix
        self.h_per_pix = h_per_pix
        self.width = width
        self.height = height
        self.ref_coords = ref_coords
        self.terms, self.num_terms = terms, num_terms
        self.breakout = breakout
        self.precise_reference = precise_reference
        self.iter_accurate = iter_accurate
        self.max_iterations = max_iterations
        self.scaling_factor = scaling_factor


class PerturbationCL(MandelbrotCL):
    def get_program_contents(self):
        with open(Path(__file__).parent / "mandelbrot_perturbations.cl") as f:
            return f.read()

    def _compute(self, pert: PerturbationState, fix_glitches):
        my_logger.debug("computing")
        real_dtype = self._get_real_dtype()
        complex_dtype = self._get_complex_dtype()

        terms_buf = cl.Buffer(
            self.ctx,
            self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
            hostbuf=pert.terms.astype(complex_dtype),
        )
        precise_ref_buf = cl.Buffer(
            self.ctx,
            self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
            hostbuf=pert.precise_reference.astype(complex_dtype),
        )

        self.prg.approximate_pixels(
            self.queue,
            self.iterations_grid.shape,
            None,
            self.ibuf,
            self.pbuf,
            real_dtype(pert.w_per_pix),
            real_dtype(pert.h_per_pix),
            np.int32(pert.width),
            np.int32(pert.ref_coords[1]),
            np.int32(pert.ref_coords[0]),
            terms_buf,
            np.int32(pert.num_terms),
            real_dtype(pert.scaling_factor),
            np.int32(pert.breakout),
            precise_ref_buf,
            np.int32(pert.iter_accurate),
            np.int32(pert.max_iterations),
            np.int32(GLITCH_ITER),
            real_dtype(GLITCH_DIFF_THRESHOLD),
            np.int32(fix_glitches),
            np.int32(BREAKOUT_R2),
        )

    def compute(
        self, pert: PerturbationState, fix_glitches: bool, double_precision: bool
    ):
        self.set_precision(double_precision)
        with self.manage_buffer(pert.height, pert.width):
            self._compute(pert, fix_glitches)
        return self.out


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

        if actual_size < GLITCH_DIFF_THRESHOLD * sq_mod(ref_i) or (
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
def approximate_pixels(
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
