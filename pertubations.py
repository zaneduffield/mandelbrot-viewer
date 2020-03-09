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
ERROR_THRESH = 0.00001
GLITCH_ITER = -1

@njit
def _iterate_series_constants(ref_hist, ref_escaped_at: int, probe_deltas_init, terms, num_terms: int):
    probe_deltas_cur = probe_deltas_init.copy()
    num_terms = len(terms)
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
                return i

    return ref_escaped_at


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


def compute_series_constants(b_left: ComplexBf, t_right: ComplexBf, ref_init: ComplexBf, ref_hist: np.ndarray, ref_escaped_at: int, iterations: int, num_terms, num_probes):
    probes = []
    square_side_len = int(math.sqrt(num_probes))
    for i in range(square_side_len):
        for j in range(square_side_len):
            x_ratio, y_ratio = i/square_side_len, j/square_side_len
            probes.append(b_left + x_ratio*(t_right.real - b_left.real) + y_ratio*(t_right.imag - b_left.imag))

    terms = np.zeros((num_terms, iterations + 1), dtype=np.complex_, order="F")
    p_deltas_init = np.array([complex(p - ref_init) for p in probes])
    accurate_iters = _iterate_series_constants(ref_hist, ref_escaped_at, p_deltas_init, terms, num_terms)
    return terms, accurate_iters


MAX_GLITCH_FIX_LOOPS = 20
NUM_SERIES_TERMS = 10
NUM_RANDOM_REFS_DESPARATE = 15
def mandelbrot_pertubation(b_left: ComplexBf, t_right: ComplexBf, height, width, iterations, num_probes, num_series_terms):
    width_per_pixel = float((t_right.real - b_left.real)/width)
    height_per_pixel = float((t_right.imag - b_left.imag)/height)
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    ref_coords = width//2, height//2
    glitched_count = 0
    prev_refs = []
    num_prev_refs = 0
    loops = 0

    def _ref_from_prop(r_prop, i_prop):
        return ComplexBf(b_left.real + r_prop * (t_right.real - b_left.real),
                             b_left.imag + i_prop * (t_right.imag - b_left.imag))

    while loops <= MAX_GLITCH_FIX_LOOPS:
        prev_refs.append(ref_coords)
        num_prev_refs += 1

        ref_r_prop = ref_coords[0] / width
        ref_i_prop = ref_coords[1] / height
        ref = _ref_from_prop(ref_r_prop, ref_i_prop)
        print("iterating reference")
        ref_hist, ref_hist_abs, ref_escaped_at = iterate_ref(ref, iterations)
        print("computing series constants...")
        terms, iter_accurate = compute_series_constants(b_left, t_right, ref, ref_hist, ref_escaped_at, iterations, NUM_SERIES_TERMS, num_probes)

        print(f"proceeding with {iter_accurate} reference iterations")
        print(f"reference broke out at {ref_escaped_at}")

        pertubation_computer = PertubationComputer(
            width_per_pixel, width, height_per_pixel, height, np.array(ref_coords), terms, num_series_terms,
            ref_escaped_at, ref_hist, ref_hist_abs, iter_accurate, iterations
        )
        if not loops:
            glitched_count = approximate_pixels(pertubation_computer, iterations_grid)
        else:
            glitched_count = approximate_pixels(pertubation_computer, iterations_grid, fix_glitches=True)
        print(f"{glitched_count} glitched pixels remaining")
        if not glitched_count:
            break
        if glitched_count:
            if glitched_count > 0.4 * height * width:
                print("iterating multiple reference and choosing best")
                best_iters = 0
                for _ in range(NUM_RANDOM_REFS_DESPARATE):
                    r_prop, i_prop = random.random(), random.random()
                    test_ref = _ref_from_prop(r_prop, i_prop)
                    if test_ref in prev_refs:
                        continue
                    _, _, ref_escaped_at = iterate_ref(test_ref, iterations)
                    if ref_escaped_at > best_iters:
                        ref_coords = round(r_prop*width), round(i_prop*height)
                        best_iters = ref_escaped_at

            else:
                # ref_coords = get_random_new_ref(iterations_grid, width, height, glitched_count)
                ref_coords = get_new_ref(iterations_grid, width, height, np.array(prev_refs), num_prev_refs)
            if ref_coords is None:
                ref_coords = random.randint(0, width), random.randint(0, height)
            print(f"new ref at :{ref_coords}")
        loops += 1

    for x in prev_refs:
        try:
            iterations_grid[x[1], x[0]] = iterations + 1
        except IndexError:
            print("debug here")

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

JUMP = 1
EXCLUDE_RADIUS_2 = 1**2
@njit
def get_new_ref(iterations_grid, width, height, exclude_refs, num_exclude_refs):
    # find center of tha largest blob
    grid = np.empty((height, width), dtype=np.int32)
    best_point = (0, 0)
    best_size = 0
    for j in range(0, height, JUMP):
        i = 0
        while i < width:
            blob_best_height_at_x = i
            blob_best_height = 0
            blob_width = 0
            if iterations_grid[j, i] == GLITCH_ITER:
                # find right edge of glitched blob
                blob_start_x = i
                blob_width = 1
                for k in range(i + 1, width):
                    if iterations_grid[j, k] != GLITCH_ITER:
                        blob_width = k - blob_start_x
                        break
                    blob_width = k - blob_start_x + 1
                    cur_x = k
                    # find top edge of glitched blob
                    blob_y_max = j
                    for k in range(j + 1, height):
                        blob_y_max = k - 1
                        if iterations_grid[k, cur_x] != GLITCH_ITER:
                            break
                    # find bottom edge of glitched blob
                    blob_y_min = j
                    for k in range(j - 1, -1, -1):
                        blob_y_min = k + 1
                        if iterations_grid[k, cur_x] != GLITCH_ITER:
                            break
                    blob_height = blob_y_max - blob_y_min + 1
                    if blob_height > blob_best_height:
                        blob_best_height_at_x = cur_x
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

            i += JUMP

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
        # TODO: why does using iter_accurate without -1 cause issues???
        delta_i = self.get_estimate(x, y, self.iter_accurate)
        delta_0 = self.get_delta(x, y)
        this_breakout = 0
        for i in range(self.iter_accurate, self.breakout):
            x_i = self.precise_reference[i]
            actual_size = abs(delta_i + x_i)
            # detect glitched pixels
            if actual_size < 0.001 * self.precise_reference_abs[i]:
                iterations_grid[y, x] = GLITCH_ITER
                return -1

            if actual_size <= BREAKOUT_R:
                delta_i = 2 * self.precise_reference[i] * delta_i + delta_i * delta_i + delta_0
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
def approximate_pixels(precision: PertubationComputer, iterations_grid, fix_glitches: bool = False):
    glitched_count = 0
    legit_glitched_count = 0

    for y in prange(precision.height):
        for x in range(precision.width):
            if fix_glitches and iterations_grid[y, x] != GLITCH_ITER:
                continue
            result = precision.approximate_pixel(x, y, iterations_grid)
            if result:
                if glitched_count == 0:
                    print(x, y)
                glitched_count += 1
                if result < 0:
                    legit_glitched_count += 1
    print("legit glitched", legit_glitched_count)
    return glitched_count
