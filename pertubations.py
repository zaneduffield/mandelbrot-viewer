import math
import random
from typing import List

import numpy as np
from numba import jit, jitclass, njit, prange
from numba import int32, float32, float64, complex128, int64
from bigfloat import BigFloat, Context, setcontext
from complex_bf import ComplexBf
from mandelbrot import BREAKOUT_R_2

BREAKOUT_R = math.sqrt(BREAKOUT_R_2)
ERROR_THRESH = 0.01
NUM_PROBES = 100
GLITCH_ITER = -1

@njit
def _iterate_series_constants(ref_hist, iterations: int, probe_deltas_init, terms, num_terms: int):
    probe_deltas_cur = probe_deltas_init.copy()
    num_terms = len(terms)
    terms[0][0] = 1
    for i in range(1, num_terms):
        terms[i][0] = 0

    error_thresh_met = False
    for i in range(iterations):
        z_comp = ref_hist[i]
        terms[0][i + 1] = 2 * z_comp * terms[0][i] + 1
        for j in range(1, num_terms):
            s = 0
            for k in range(j):
                s += terms[k][i] * terms[j - k - 1][i]
            terms[j][i + 1] = 2 * z_comp * terms[j][i] + s

        for j in range(len(probe_deltas_init)):
            delta = probe_deltas_cur[j]
            probe_deltas_cur[j] = 2 * z_comp * delta + delta * delta + probe_deltas_init[j]

            z_del_app = 0
            delta = probe_deltas_init[j]
            for k in range(num_terms):
                z_del_app += terms[k][i + 1] * delta
                delta *= probe_deltas_init[j]
            if abs(probe_deltas_cur[j] - z_del_app) > ERROR_THRESH:
                error_thresh_met = True
                break

        if error_thresh_met:
            break

    return i


def iterate_ref(ref: ComplexBf, iterations):
    ref_hist = np.empty(iterations, dtype=np.complex_)
    ref_escaped_at = 0
    ref_curr = ref
    for i in range(iterations):
        ref_hist[i] = complex(ref_curr)
        ref_curr = ref_curr*ref_curr + ref
        if not ref_escaped_at and ref_curr.abs_2() > BREAKOUT_R_2:
            ref_escaped_at = i + 1

    return ref_curr, ref_hist, ref_escaped_at


def compute_series_constants(b_left: ComplexBf, t_right: ComplexBf, ref_init, ref_hist, iterations: int, num_terms, num_probes):
    probes = []
    square_side_len = int(math.sqrt(num_probes))
    for i in range(square_side_len):
        for j in range(square_side_len):
            x_ratio, y_ratio = i/square_side_len, j/square_side_len
            probes.append(b_left + x_ratio*(t_right.real - b_left.real) + y_ratio*(t_right.imag - b_left.imag))

    terms = np.empty((num_terms, iterations + 1), dtype=np.complex_, order="F")
    p_deltas_init = np.array([complex(p - ref_init) for p in probes])
    accurate_iters = _iterate_series_constants(ref_hist, iterations, p_deltas_init, terms, num_terms)
    return terms, accurate_iters


@njit
def myave(ary, n):
    avg_x = avg_y = 0
    t = 1
    for i in range(n):
        avg_x += (ary[i][0] - avg_x) / t
        avg_y += (ary[i][1] - avg_y) / t
        t += 1
    return avg_x, avg_y

MAX_GLITCH_FIX_LOOPS = 5
NUM_SERIES_TERMS = 10
REF_GRID_WIDTH = 5
def mandelbrot_pertubation(b_left: ComplexBf, t_right: ComplexBf, height, width, iterations, num_probes, num_series_terms):
    width_per_pixel = float(t_right.real - b_left.real)/width
    height_per_pixel = float(t_right.imag - b_left.imag)/height
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    z_hist_abs = np.empty(iterations, dtype=np.double)
    ref_coords = width//2, height//2
    glitched_count = 0
    prev_refs = []
    num_prev_refs = 0
    loops = 0

    def _ref_from_prop(r_prop, i_prop):
        return ComplexBf(b_left.real + r_prop * (t_right.real - b_left.real),
                             b_left.imag + i_prop * (t_right.imag - b_left.imag))

    while loops < MAX_GLITCH_FIX_LOOPS:
        print("computing series constants...")
        prev_refs.append(ref_coords)
        num_prev_refs += 1
        ref_r_prop = ref_coords[0] / width
        ref_i_prop = ref_coords[1] / height
        ref = _ref_from_prop(ref_r_prop, ref_i_prop)

        ref_curr, ref_hist, ref_escaped_at = iterate_ref(ref, iterations)
        terms, iter_accurate = compute_series_constants(b_left, t_right, ref, ref_hist, iterations, NUM_SERIES_TERMS, num_probes)

        print(f"proceeding with {iter_accurate} reference iterations")
        breakout = ref_escaped_at or iterations
        print(f"reference broke out at {breakout}")

        iterations_computer = Precision(width_per_pixel, width, height_per_pixel, height, np.array(ref_coords), terms, num_series_terms, breakout,
                                   ref_hist, z_hist_abs, iter_accurate, iterations)
        if not glitched_count:
            glitched_count = approximate_pixels(iterations_computer, iterations_grid)
        else:
            glitched_count = approximate_pixels(iterations_computer, iterations_grid, fix_glitches=True)

        print(f"{glitched_count} glitched pixels remaining")
        if not glitched_count:
            break
        else:
            if glitched_count > 0.3 * height * width:
                print("iterating multiple reference and choosing best")
                best_iters = 0
                for i in range(REF_GRID_WIDTH):
                    for j in range(REF_GRID_WIDTH):
                        test_ref = _ref_from_prop(i/REF_GRID_WIDTH, j/REF_GRID_WIDTH)
                        _, __, ref_escaped_at = iterate_ref(test_ref, iterations)
                        if ref_escaped_at > best_iters:
                            ref_coords = i*width//REF_GRID_WIDTH, j*height//REF_GRID_WIDTH
                            best_iters = ref_escaped_at

                # ref_coords = get_random_new_ref(iterations_grid, width, height, glitched_count)
            else:
                ref_coords = get_new_ref(iterations_grid, width, height, np.array(prev_refs), num_prev_refs)
            if ref_coords is None:
                ref_coords = random.randint(0, width), random.randint(0, height)
            print(f"new ref at :{ref_coords}")

        loops += 1

    for x in prev_refs:
        iterations_grid[x[1], x[0]] = iterations + 1

    return iterations_grid

@njit
def get_random_new_ref(iterations_grid, width, height, glitched_count):
    x = random.randint(0, glitched_count)
    for j in range(height):
        for i in range(width):
            if iterations_grid[j, i] == GLITCH_ITER:
                if x == 0:
                    return i, j
                x -= 1

JUMP = 5
EXCLUDE_RADIUS_2 = 25**2
@njit
def get_new_ref(iterations_grid, width, height, exclude_refs, num_exclude_refs):
    # find center of tha largest blob
    best_point = (0, 0)
    best_size = 0
    for j in range(0, height):
        row_best_point = (0, 0)
        row_best_size = 0
        i = 0
        while i < width:
            i += 1
            if iterations_grid[j, i] != GLITCH_ITER:
                continue
            # find right edge of glitched blob
            for k in range(i + 1, width):
                if iterations_grid[j, k] != GLITCH_ITER:
                    break
                blob_x = k
                blob_width = k - i
                # find bottom edge of glitched blob
                for k in range(j + 1, height):
                    if iterations_grid[k, blob_x] != GLITCH_ITER:
                        break
                blob_y = (k + j)//2
                blob_height = k - j
                # blob_height, mid_y = 0, j
                if blob_width + blob_height > row_best_size:
                    row_best_point = blob_x, blob_y
                    row_best_size = blob_width + blob_height

                i += 1

        if row_best_size > best_size:
            excluded = False
            for m in range(num_exclude_refs):
                ref = exclude_refs[m]
                delta = ref[0] - row_best_point[0], ref[1] - row_best_point[1]
                if delta[0]*delta[0] + delta[1]*delta[1] < EXCLUDE_RADIUS_2:
                    excluded = True
                    break
            if not excluded:
                best_point = row_best_point
                best_size = row_best_size

    return best_point


spec = [
    ('w_per_pix', float64),
    ('h_per_pix', float64),
    ('width', int64),
    ('height', int64),
    ('ref_coords', int32[:]),
    ('terms', complex128[:,:]),
    ('num_terms', int64),
    ('breakout', int64),
    ('precise_reference', complex128[:]),
    ('precise_reference_abs', float64[:]),
    ('iter_accurate', int64),
    ('iterations', int64),
    ('init_deltas_iter', complex128[:, :]),
    ('deltas_iter', complex128[:, :])
]

@jitclass(spec)
class Precision:
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

        self.init_deltas_iter = np.empty((self.height, self.width), dtype=np.complex_)
        self.deltas_iter = np.empty((self.height, self.width), dtype=np.complex_)
        for j in range(self.height):
            for i in range(self.width):
                self.init_deltas_iter[j, i] = self.get_estimate(i, j, self.iter_accurate - 1)

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
        ver_delta = (j - self.ref_coords[1]) * self.h_per_pix
        return complex(hor_delta, ver_delta)

    def get_estimate(self, i, j, iteration):
        init_delta = self.get_delta(i, j)
        delta = init_delta
        out = 0
        for k in range(self.num_terms):
            out += self.terms[k][iteration] * delta
            delta *= init_delta
        return out

    def approximate_pixel(self, x, y, iterations_grid):
        self.deltas_iter[y, x] = self.get_estimate(x, y, self.iter_accurate - 1)
        delta_0 = self.get_delta(x, y)
        for i in range(self.iter_accurate, self.breakout):
            delta_i = self.deltas_iter[y, x]
            x_i = self.precise_reference[i - 1]
            actual_size = abs(delta_i + x_i)
            # detect glitched pixels
            if actual_size < 0.001 * self.precise_reference_abs[i-1]:
                iterations_grid[y, x] = GLITCH_ITER
                return -1

            if actual_size <= BREAKOUT_R:
                self.deltas_iter[y, x] = 2 * self.precise_reference[i - 1] * delta_i + delta_i * delta_i + delta_0
            else:
                break

        if i == self.breakout - 1:
            if self.breakout < self.iterations:
                iterations_grid[y, x] = GLITCH_ITER
                return 1
            else:
                iterations_grid[y, x] = 0
        elif i == self.iter_accurate:
            # if broke out before iterating, find true breakout value using binary search on accurate estimations
            iterations_grid[y, x] = self.search_for_escape(0, self.iter_accurate, x, y) + 1
        else:
            iterations_grid[y, x] = i

        return 0


@njit(parallel=True, fastmath=True)
def approximate_pixels(precision: Precision, iterations_grid, fix_glitches: bool = False):
    glitched_count = 0
    legit_glitched_count = 0

    for j in prange(precision.height):
        for i in range(precision.width):
            if fix_glitches and iterations_grid[j, i] != GLITCH_ITER:
                continue
            result = precision.approximate_pixel(i, j, iterations_grid)
            if result:
                glitched_count += 1
                if result < 0:
                    legit_glitched_count += 1
    print("legit glitched", legit_glitched_count)
    return glitched_count
