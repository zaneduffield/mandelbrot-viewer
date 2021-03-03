import math
import random
import time
import numpy as np
import pyopencl as cl
import pyopencl.tools

from numba import njit, prange, float64, complex128, int32
from numba.experimental import jitclass

from constants import GLITCH_ITER, MAX_GLITCH_FIX_LOOPS, MAX_OK_GLITCH_COUNT
from mandelbrot import BREAKOUT_R_2
from gmpy2 import mpc

from opencl_test import MandelbrotCL


def iterate_ref(ref: mpc, iterations):
    ref_hist = np.empty(iterations, dtype=np.complex_)
    ref_curr = ref
    for i in range(iterations):
        temp = complex(ref_curr)
        temp_abs = temp.real * temp.real + temp.imag * temp.imag
        if temp_abs > BREAKOUT_R_2:
            return ref_hist, i

        ref_hist[i] = temp
        ref_curr = ref_curr * ref_curr + ref

    return ref_hist, iterations


@njit
def iterate_series_constants(ref_hist, ref_escaped_at: int, probe_deltas_init, terms, num_terms: int, error_threshold: float):
    probe_deltas_cur = probe_deltas_init.copy()

    terms[0][0] = 1
    for i in range(1, num_terms):
        terms[i][0] = 0

    for i in range(ref_escaped_at):
        z_comp = ref_hist[i]
        terms[0][i + 1] = 2 * z_comp * terms[0][i] + 1
        for j in range(1, num_terms):
            s = 0
            for k in range(j):
                s += terms[k][i] * terms[j - k - 1][i]
                if np.isnan(s):
                    break
            terms[j][i + 1] = 2 * z_comp * terms[j][i] + s

        for j in range(len(probe_deltas_init)):
            delta = probe_deltas_cur[j]
            probe_deltas_cur[j] = 2 * z_comp * delta + delta * delta + probe_deltas_init[j]

            z_del_app = 0
            delta = probe_deltas_init[j]
            for k in range(num_terms):
                term = terms[k][i + 1]
                if np.isnan(term):
                    break
                z_del_app += term * delta
                delta *= probe_deltas_init[j]

            diff = probe_deltas_cur[j] - z_del_app
            if diff.real * diff.real + diff.imag * diff.imag > error_threshold:
                return i

    return ref_escaped_at


def get_probe_deltas(t_left: mpc, b_right: mpc, ref_init: mpc, num_probes):
    probes = []
    square_side_len = int(math.sqrt(num_probes))
    for i in range(square_side_len):
        for j in range(square_side_len):
            x_ratio, y_ratio = i / square_side_len, j / square_side_len
            probe_del = x_ratio * (b_right.real - t_left.real) - y_ratio * (t_left.imag - b_right.imag) * 1j
            probes.append(complex(t_left + probe_del - ref_init))

    return np.array(probes)


def compute_series_constants(t_left: mpc, b_right: mpc, ref_init: mpc, ref_hist: np.ndarray, ref_escaped_at: int,
                             iterations: int, num_terms, num_probes):
    p_deltas_init = get_probe_deltas(t_left, b_right, ref_init, num_probes)
    terms = np.zeros((num_terms, iterations + 1), dtype=np.complex_, order="F")

    error_threshold = float((b_right.real - t_left.real))**2
    accurate_iters = iterate_series_constants(ref_hist, ref_escaped_at, p_deltas_init, terms, num_terms, error_threshold)
    return terms, accurate_iters



class PerturbationComputer:
    def __init__(self):
        self.cl = None

    def compute(self, t_left: mpc, b_right: mpc, height, width, iterations, num_probes, num_series_terms, gpu: bool):
        width_per_pixel = float((b_right.real - t_left.real) / width)
        iterations_grid = np.zeros((height, width), dtype=np.int32)
        ref_coords = width // 2, height // 2

        def _ref_from_coords(coords):
            return t_left + mpc(coords[0] * width_per_pixel - coords[1] * width_per_pixel * 1j)

        if gpu and self.cl is None:
            self.cl = PerturbationCL()

        loops = 0
        while loops <= MAX_GLITCH_FIX_LOOPS:
            ref = _ref_from_coords(ref_coords)
            print("iterating reference")
            ref_hist, ref_escaped_at = iterate_ref(ref, iterations)
            print("computing series constants...")
            terms, iter_accurate = compute_series_constants(t_left, b_right, ref, ref_hist, ref_escaped_at, iterations,
                                                            num_series_terms, num_probes)

            print(f"proceeding with {iter_accurate} reference iterations")
            print(f"reference broke out at {ref_escaped_at}")

            pertubation_state = PertubationState(
                width_per_pixel, width, height, np.array(ref_coords, dtype=np.int32), terms, num_series_terms,
                ref_escaped_at, ref_hist, iter_accurate, iterations
            )

            if gpu:
                iterations_grid = self.cl.get_pixels(pertubation_state, fix_glitches=loops)
                glitched_count = get_glitched_count(pertubation_state, iterations_grid)
            else:
                glitched_count = approximate_pixels(pertubation_state, iterations_grid, fix_glitches=loops)

            yield iterations_grid
            print(f"{glitched_count} glitched pixels remaining")
            if glitched_count <= MAX_OK_GLITCH_COUNT:
                break

            ref_coords = get_new_ref(iterations_grid, width, height, GLITCH_ITER)
            if ref_coords is None:
                ref_coords = random.randint(0, width), random.randint(0, height)
            print(f"new ref at :{ref_coords}")
            loops += 1
        #
        # # # to see the reference pixels in the image
        # # for x in prev_refs[1:]:
        # #     iterations_grid[x[1], x[0]] = iterations + 1
        #
        #


@njit
def get_random_new_ref(iterations_grid, width, height, glitched_count):
    x = random.randint(0, glitched_count - 1)
    for j in range(height):
        for i in range(width):
            if iterations_grid[j, i] == GLITCH_ITER:
                if x == 0:
                    return i, j
                x -= 1


@njit(parallel=True)
def get_new_ref(iterations_grid, width, height, blob_iter):
    lo_x, hi_x = 0, width
    lo_y, hi_y = 0, height

    blob_counts = [0, 0]
    while max(blob_counts) / ((hi_x - lo_x) * (hi_y - lo_y)) < 0.499:
        blob_counts = [0, 0]

        if hi_x - lo_x > hi_y - lo_y:
            mid = (hi_x + lo_x) // 2
            for y in prange(lo_y, hi_y):
                for x in range(lo_x, mid):
                    if iterations_grid[y, x] == blob_iter:
                        blob_counts[0] += 1

                for x in range(mid, hi_x):
                    if iterations_grid[y, x] == blob_iter:
                        blob_counts[1] += 1

            if blob_counts[0] > blob_counts[1]:
                hi_x = mid
            else:
                lo_x = mid
        else:
            mid = (hi_y + lo_y) // 2
            for x in prange(lo_x, hi_x):
                for y in range(lo_y, mid):
                    if iterations_grid[y, x] == blob_iter:
                        blob_counts[0] += 1

                for y in range(mid, hi_y):
                    if iterations_grid[y, x] == blob_iter:
                        blob_counts[1] += 1

            if blob_counts[0] > blob_counts[1]:
                hi_y = mid
            else:
                lo_y = mid

    return (hi_x + lo_x) // 2, (hi_y + lo_y) // 2


spec = [
    ('w_per_pix', float64),
    ('width', int32),
    ('height', int32),
    ('ref_coords', int32[:]),
    ('terms', complex128[:, :]),
    ('num_terms', int32),
    ('breakout', int32),
    ('precise_reference', complex128[:]),
    ('iter_accurate', int32),
    ('iterations', int32),
]


@jitclass(spec)
class PertubationState:
    def __init__(self, w_per_pix, width, height, ref_coords, terms, num_terms, breakout,
                 precise_reference, iter_accurate, iterations):
        self.w_per_pix = w_per_pix
        self.width = width
        self.height = height
        self.ref_coords = ref_coords
        self.terms, self.num_terms = terms, num_terms
        self.breakout = breakout
        self.precise_reference = precise_reference
        self.iter_accurate = iter_accurate
        self.iterations = iterations


class PerturbationCL(MandelbrotCL):
    def __init__(self):
        super().__init__()
        with open('mandelbrot_perturbations.cl') as f:
            super().compile(f.read())

    def _compute_pixels(self, pert, fix_glitches):
        print("computing")
        terms_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=pert.terms)
        precise_ref_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=pert.precise_reference)

        self.prg.approximate_pixels(self.queue, self.a.shape, None, self.abuf,
                                   np.float64(pert.w_per_pix),
                                   np.int32(pert.width),
                                   np.int32(pert.ref_coords[0]),
                                   np.int32(pert.ref_coords[1]),
                                   terms_buf,
                                   np.int32(pert.num_terms),
                                   np.int32(pert.breakout),
                                   precise_ref_buf,
                                   np.int32(pert.iter_accurate),
                                   np.int32(pert.iterations),
                                   np.int32(GLITCH_ITER),
                                   np.int32(fix_glitches),
                                   )

    def get_pixels(self, pert, fix_glitches):
        with self.manage_buffer(pert.height, pert.width):
            self._compute_pixels(pert, fix_glitches)
        return self.out


@njit
def search_for_escape(data, lo, hi, x, y):
    while hi != lo:
        mid = (lo + hi) // 2
        point = data.precise_reference[mid] + get_estimate(data, x, y, mid)
        if point.real * point.real + point.imag * point.imag <= BREAKOUT_R_2:
            lo = mid + 1
        else:
            hi = mid
    return hi


@njit
def get_delta(data, i, j):
    hor_delta = (i - data.ref_coords[0]) * data.w_per_pix
    ver_delta = (data.ref_coords[1] - j) * data.w_per_pix
    return complex(hor_delta, ver_delta)


@njit
def get_estimate(data, i, j, iteration):
    init_delta = get_delta(data, i, j)
    delta = init_delta
    out = 0
    for k in range(data.num_terms):
        term = data.terms[k][iteration]
        out += term * delta
        delta *= init_delta
    return out


@njit
def approximate_pixel(data, x, y, iterations_grid):
    # TODO: why does using iter_accurate without -1 cause issues???
    delta_i = get_estimate(data, x, y, data.iter_accurate - 1)
    delta_0 = get_delta(data, x, y)
    this_breakout = 0
    for i in range(data.iter_accurate, data.breakout):
        x_i = data.precise_reference[i - 1]
        point = delta_i + x_i
        actual_size = point.real * point.real + point.imag * point.imag

        if actual_size < 0.000001 * (x_i.real * x_i.real + x_i.imag * x_i.imag):
            iterations_grid[y, x] = GLITCH_ITER
            return -1

        if actual_size <= BREAKOUT_R_2:
            delta_i = 2 * data.precise_reference[i - 1] * delta_i + delta_i * delta_i + delta_0
        else:
            break
        this_breakout = i + 1

    if this_breakout == 0:
        # broke out before iterating, find true breakout value using binary search on accurate estimations
        iterations_grid[y, x] = search_for_escape(data, 0, data.iter_accurate, x, y) + 1
    elif this_breakout == data.breakout:
        if data.breakout < data.iterations:
            iterations_grid[y, x] = GLITCH_ITER
            return 1
        else:
            iterations_grid[y, x] = 0
    else:
        iterations_grid[y, x] = this_breakout

    return 0


@njit(parallel=True)
def approximate_pixels(precision: PertubationState, iterations_grid, fix_glitches):
    glitched_count = 0
    legit_glitched_count = 0

    for y in prange(precision.height):
        for x in range(precision.width):
            if fix_glitches and iterations_grid[y, x] != GLITCH_ITER:
                continue
            result = approximate_pixel(precision, x, y, iterations_grid)
            if result:
                glitched_count += 1
                if result < 0:
                    legit_glitched_count += 1
    print("legit glitched", legit_glitched_count)
    return glitched_count


@njit
def get_glitched_count(pert: PertubationState, iterations_grid):
    count = 0
    for y in range(pert.height):
        for x in range(pert.width):
            if iterations_grid[y, x] == GLITCH_ITER:
                count += 1

    return count
