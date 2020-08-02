import math
import random
from timeit import timeit
from typing import List

import numpy as np
from numba import njit, prange, float64, complex128, int64
from numba.experimental import jitclass
from bigfloat import BigFloat
from complex_bf import ComplexBf
from mandelbrot import BREAKOUT_R_2
BREAKOUT_R = math.sqrt(BREAKOUT_R_2)
ERROR_THRESH = 0.0001
GLITCH_ITER = -1


def iterate_ref(ref: ComplexBf, iterations):
    ref_hist = np.zeros(iterations, dtype=np.complex_)
    ref_hist_abs = np.zeros(iterations, dtype=np.float64)
    ref_curr = ref
    for i in range(iterations):
        if ref_curr.abs_2() > BREAKOUT_R_2:
            return ref_hist, ref_hist_abs, i
        temp = complex(ref_curr)
        ref_hist[i] = temp
        ref_hist_abs[i] = abs(temp)
        ref_curr = ref_curr*ref_curr + ref

    return ref_hist, ref_hist_abs, iterations


@njit
def iterate_series_constants(ref_hist, ref_escaped_at: int, probe_deltas_init, terms, num_terms: int):
    probe_deltas_cur = probe_deltas_init.copy()
    probe_deltas_scale = 0
    for p in probe_deltas_init:
        if abs(p) > probe_deltas_scale:
            probe_deltas_scale = abs(p)

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
            if abs(probe_deltas_cur[j] - z_del_app) / abs(z_del_app) > ERROR_THRESH:
                return i

    return ref_escaped_at


def get_probe_deltas(t_left: ComplexBf, b_right: ComplexBf, ref_init: ComplexBf, num_probes):
    probes = []
    square_side_len = int(math.sqrt(num_probes))
    for i in range(square_side_len):
        for j in range(square_side_len):
            x_ratio, y_ratio = i / square_side_len, j / square_side_len
            probe_del = ComplexBf(BigFloat(x_ratio * (b_right.real - t_left.real)),
                                  BigFloat(-y_ratio * (t_left.imag - b_right.imag)))
            probes.append(t_left + probe_del)

    return np.array([complex(p - ref_init) for p in probes])


def compute_series_constants(t_left: ComplexBf, b_right: ComplexBf, ref_init: ComplexBf, ref_hist: np.ndarray, ref_escaped_at: int, iterations: int, num_terms, num_probes):
    p_deltas_init = get_probe_deltas(t_left, b_right, ref_init, num_probes)
    terms = np.zeros((num_terms, iterations + 1), dtype=np.complex_, order="F")
    accurate_iters = iterate_series_constants(ref_hist, ref_escaped_at, p_deltas_init, terms, num_terms)
    return terms, accurate_iters


MAX_GLITCH_FIX_LOOPS = 5
NUM_RANDOM_REFS_DESPARATE = 15
MAX_GLITCH_COUNT = 100
def mandelbrot_pertubation(t_left: ComplexBf, b_right: ComplexBf, height, width, iterations, num_probes, num_series_terms):
    width_per_pixel = float((b_right.real - t_left.real)/width)
    height_per_pixel = float((t_left.imag - b_right.imag)/height)
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    ref_coords = width//2, height//2
    prev_refs = []
    num_prev_refs = 0
    loops = 0

    def _ref_from_coords(coords):
        return ComplexBf(t_left.real + coords[0]*width_per_pixel, t_left.imag - coords[1]*height_per_pixel)

    while loops <= MAX_GLITCH_FIX_LOOPS:
        prev_refs.append(ref_coords)
        num_prev_refs += 1

        ref = _ref_from_coords(ref_coords)
        print("iterating reference")
        ref_hist, ref_hist_abs, ref_escaped_at = iterate_ref(ref, iterations)
        print("computing series constants...")
        terms, iter_accurate = compute_series_constants(t_left, b_right, ref, ref_hist, ref_escaped_at, iterations, num_series_terms, num_probes)

        print(f"proceeding with {iter_accurate} reference iterations")
        print(f"reference broke out at {ref_escaped_at}")

        pertubation_computer = PertubationComputer(
            width_per_pixel, width, height_per_pixel, height, np.array(ref_coords, dtype=np.int64), terms, num_series_terms,
            ref_escaped_at, ref_hist, ref_hist_abs, iter_accurate, iterations
        )
        glitched_count = approximate_pixels(pertubation_computer, iterations_grid, fix_glitches=loops)
        print(f"{glitched_count} glitched pixels remaining")
        if glitched_count <= MAX_GLITCH_COUNT:
            break

        ref_coords = get_new_ref(iterations_grid, width, height, np.array(prev_refs), num_prev_refs, GLITCH_ITER)
        if ref_coords is None:
            ref_coords = random.randint(0, width), random.randint(0, height)
        print(f"new ref at :{ref_coords}")
        loops += 1

    # # to see the reference pixels in the image
    # for x in prev_refs[1:]:
    #     iterations_grid[x[1], x[0]] = iterations + 1

    return iterations_grid


@njit
def get_random_new_ref(iterations_grid, width, height, glitched_count):
    x = random.randint(0, glitched_count - 1)
    for j in range(height):
        for i in range(width):
            if iterations_grid[j, i] == GLITCH_ITER:
                if x == 0:
                    return i, j
                x -= 1

EXCLUDE_RADIUS_2 = 1
@njit
def get_new_ref(iterations_grid, width, height, exclude_refs, num_exclude_refs, blob_iter):
    # store the number of contiguous pixels with blob_iter above and below each position in the grid
    # in that order
    grid = np.zeros((height, width, 2), dtype=np.int32)
    best_point = (0, 0)
    best_size = 0
    for j in range(0, height):
        i = 0
        while i < width:
            blob_best_height_at_x = i
            blob_best_height = 0
            blob_width = 0
            # find right edge of glitched blob
            blob_start_x = i
            for k in range(i, width):
                if iterations_grid[j, k] != blob_iter:
                    blob_width = k - blob_start_x
                    break

                if j + 1 < height:
                    grid[j + 1, i][1] = grid[j, i][1] + 1

                blob_width = k - blob_start_x + 1
                # find top edge of glitched blob
                if j > 0 and grid[j - 1, i][0]:
                    blob_y_max = j + grid[j - 1, i][0] - 1
                else:
                    blob_y_max = j
                    for k in range(j + 1, height):
                        blob_y_max = k - 1
                        if iterations_grid[k, i] != blob_iter:
                            break
                        grid[j, i][0] += 1
                # find bottom edge of glitched blob
                blob_y_min = j - grid[j, i][1]
                # for k in range(j - 1, -1, -1):
                #     blob_y_min = k + 1
                #     if iterations_grid[k, i] != blob_iter:
                #         break
                blob_height = blob_y_max - blob_y_min + 1
                if blob_height > blob_best_height:
                    blob_best_height_at_x = i
                    blob_best_height = blob_height
                i += 1

            if blob_best_height + blob_width > best_size:
                excluded = False
                for m in range(num_exclude_refs):
                    ref = exclude_refs[m]
                    delta = ref[0] - blob_best_height_at_x, ref[1] - j
                    if delta[0]*delta[0] + delta[1]*delta[1] < EXCLUDE_RADIUS_2:
                        excluded = True
                        break
                if not excluded:
                    best_point = blob_best_height_at_x, j
                    best_size = blob_width + blob_best_height

            i += 1

    return best_point


spec = [
    ('w_per_pix', float64),
    ('h_per_pix', float64),
    ('width', int64),
    ('height', int64),
    ('ref_coords', int64[:]),
    ('terms', complex128[:,:]),
    ('num_terms', int64),
    ('breakout', int64),
    ('precise_reference', complex128[:]),
    ('precise_reference_abs', float64[:]),
    ('iter_accurate', int64),
    ('iterations', int64),
    # ('init_deltas_iter', complex128[:, :]),
    ('deltas_iter', complex128[:, :])
]

@jitclass(spec)
class PertubationComputer:
    def __init__(self, w_per_pix, width, h_per_pix, height, ref_coords, terms, num_terms, breakout,
                        precise_reference, precise_reference_abs, iter_accurate, iterations):
        self.w_per_pix = w_per_pix
        self.h_per_pix = h_per_pix
        self.width = width
        self.height = height
        self.ref_coords = ref_coords
        self.terms, self.num_terms = terms, num_terms
        self.breakout = breakout
        self.precise_reference = precise_reference
        self.precise_reference_abs = precise_reference_abs
        self.iter_accurate = iter_accurate
        self.iterations = iterations

    def search_for_escape(self, lo, hi, x, y):
        while hi != lo:
            mid = (lo + hi) // 2
            delta_estimate = self.get_estimate(x, y, mid)
            if abs(delta_estimate + self.precise_reference[mid]) <= BREAKOUT_R:
                lo = mid + 1
            else:
                hi = mid
        return hi

    def get_delta(self, i, j):
        hor_delta = (i - self.ref_coords[0]) * self.w_per_pix
        ver_delta = (self.ref_coords[1] - j) * self.h_per_pix
        return complex(hor_delta, ver_delta)

    def get_estimate(self, i, j, iteration):
        init_delta = self.get_delta(i, j)
        delta = init_delta
        out = 0
        for k in range(self.num_terms):
            term = self.terms[k][iteration]
            if np.isnan(term):
                break
            out += term * delta
            delta *= init_delta
        return out

    def approximate_pixel(self, x, y, iterations_grid):
        # TODO: why does using iter_accurate without -1 cause issues???
        delta_i = self.get_estimate(x, y, self.iter_accurate-1)
        delta_0 = self.get_delta(x, y)
        this_breakout = 0
        for i in range(self.iter_accurate, self.breakout):
            x_i = self.precise_reference[i-1]
            actual_size = abs(delta_i + x_i)
            # detect glitched pixels
            if actual_size < 0.001 * self.precise_reference_abs[i-1]:
                iterations_grid[y, x] = GLITCH_ITER
                return -1

            if actual_size <= BREAKOUT_R:
                delta_i = 2 * self.precise_reference[i-1] * delta_i + delta_i * delta_i + delta_0
            else:
                break
            this_breakout = i + 1

        if this_breakout == 0:
            # broke out before iterating, find true breakout value using binary search on accurate estimations
            iterations_grid[y, x] = self.search_for_escape(0, self.iter_accurate, x, y) + 1
        elif this_breakout == self.breakout:
            if self.breakout < self.iterations:
                iterations_grid[y, x] = GLITCH_ITER
                return 1
            else:
                iterations_grid[y, x] = 0
        else:
            iterations_grid[y, x] = this_breakout

        return 0


@njit(parallel=True)
def approximate_pixels(precision: PertubationComputer, iterations_grid, fix_glitches):
    glitched_count = 0
    legit_glitched_count = 0

    for y in prange(precision.height):
        for x in range(precision.width):
            if fix_glitches and iterations_grid[y, x] != GLITCH_ITER:
                continue
            result = precision.approximate_pixel(x, y, iterations_grid)
            if result:
                glitched_count += 1
                if result < 0:
                    legit_glitched_count += 1
    print("legit glitched", legit_glitched_count)
    return glitched_count
