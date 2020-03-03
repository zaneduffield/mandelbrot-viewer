import math
import random
from typing import List

import numpy as np
from numba import jit, jitclass, njit, prange
from numba import int32, float32, float64, complex128, int64
from bigfloat import BigFloat, Context, setcontext
from complex_bf import ComplexBf
from iterate import iterate, BREAKOUT_R_2

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


def compute_series_constants(b_left: ComplexBf, t_right: ComplexBf, seed_x_ratio, seed_y_ratio, iterations: int, num_terms, num_probes):
    ref_hist = np.empty(iterations, dtype=np.complex_)
    ref_init = ComplexBf(b_left.real + seed_x_ratio*(t_right.real - b_left.real), b_left.imag + seed_y_ratio*(t_right.imag - b_left.imag))
    ref_curr = ref_init

    z_escaped_at = 0
    for i in range(iterations):
        ref_hist[i] = complex(ref_curr)
        ref_curr = ref_curr*ref_curr + ref_init
        if not z_escaped_at and ref_curr.abs_2() > BREAKOUT_R_2:
            z_escaped_at = i + 1

    probes = []
    square_side_len = int(math.sqrt(num_probes))
    for i in range(square_side_len):
        for j in range(square_side_len):
            x_ratio, y_ratio = i/square_side_len, j/square_side_len
            probes.append(b_left + x_ratio*(t_right.real - b_left.real) + y_ratio*(t_right.imag - b_left.imag))

    terms = np.empty((num_terms, iterations + 1), dtype=np.complex_, order="F")
    p_deltas_init = np.array([complex(p - ref_init) for p in probes])
    accurate_iters = _iterate_series_constants(ref_hist, iterations, p_deltas_init, terms, num_terms)
    return terms, ref_hist, ref_curr, ref_init, accurate_iters, z_escaped_at


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
def mandelbrot_pertubation(b_left: ComplexBf, t_right: ComplexBf, height, width, iterations, num_probes, num_series_terms):
    width_per_pixel = float(t_right.real - b_left.real)/width
    height_per_pixel = float(t_right.imag - b_left.imag)/height
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    z_hist_abs = np.empty(iterations, dtype=np.double)
    ref = width//2, height//2
    prev_glitched_count = glitched_count = 0
    prev_refs = []
    num_prev_refs = 0
    loops = 0

    while loops < MAX_GLITCH_FIX_LOOPS:
        print("computing series constants...")
        prev_refs.append(ref)
        num_prev_refs += 1
        ref_r_prop = ref[0] / width
        ref_i_prop = ref[1] / height
        terms, z_hist, z, x, iter_accurate, z_escaped_at = compute_series_constants(b_left, t_right, ref_r_prop, ref_i_prop, iterations, NUM_SERIES_TERMS, num_probes)

        print(f"proceeding with {iter_accurate} reference iterations")
        breakout = z_escaped_at or iterations
        print(f"reference broke out at {breakout}")


        ref_coords = np.array([int((x.real - b_left.real)/width_per_pixel), int((x.imag - b_left.imag)/height_per_pixel)])
        iterations_computer = Precision(width_per_pixel, width, height_per_pixel, height, ref_coords, terms, num_series_terms, breakout,
                                   z_hist, z_hist_abs, iter_accurate, iterations)
        if not glitched_count:
            glitched_count = approximate_pixels(iterations_computer, iterations_grid)
        else:
            glitched_count = approximate_pixels(iterations_computer, iterations_grid, fix_glitches=True)

        print(f"{glitched_count} glitched pixels remaining")
        if not glitched_count:
            break
        else:
            ref = get_new_ref(iterations_grid, width, height, np.array(prev_refs), num_prev_refs)
            if ref is None:
                ref = random.randint(0, width), random.randint(0, height)
            print(f"new ref at :{ref}")

        loops += 1

    # if glitched_count:
    #     for i in range(glitched_count):
    #         pix = glitched_pixels[i]
    #         iterations_grid[pix[1], pix[0]] = iterations + 1
    return iterations_grid


@njit
def get_new_ref(iterations_grid, width, height, exclude_refs, num_exclude_refs):
    # find center of tha largest blob
    best_point = None
    best_size = 0
    for j in prange(height):
        i = 0
        while i < width:
            if iterations_grid[j, i] != GLITCH_ITER:
                i += 1
                continue
            # find right edge of glitched blob
            for k in range(i + 1, width):
                if iterations_grid[j, k] != GLITCH_ITER:
                    break
            mid_x = (k + i)//2
            width = k - i
            i = k
            # find bottom edge of glitched blob
            for k in range(j + 1, height):
                if iterations_grid[k, mid_x] != GLITCH_ITER:
                    break
            mid_y = (k + j)//2
            height = k - j
            if width + height > best_size:
                excluded = False
                for m in range(num_exclude_refs):
                    ref = exclude_refs[m]
                    if ref[0] == mid_x and ref[1] == mid_y:
                        excluded = True
                        break
                if not excluded:
                    best_point = mid_x, mid_y
                    best_size = width + height
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
                return False

            if actual_size <= BREAKOUT_R:
                self.deltas_iter[y, x] = 2 * self.precise_reference[i - 1] * delta_i + delta_i * delta_i + delta_0
            else:
                break

        if i == self.breakout - 1:
            if self.breakout < self.iterations:
                iterations_grid[y, x] = GLITCH_ITER
                return False
            else:
                iterations_grid[y, x] = 0
        elif i == self.iter_accurate:
            # if broke out before iterating, find true breakout value using binary search on accurate estimations
            iterations_grid[y, x] = self.search_for_escape(0, self.iter_accurate, x, y) + 1
        else:
            iterations_grid[y, x] = i

        return True


@njit(parallel=True, fastmath=True)
def approximate_pixels(precision: Precision, iterations_grid, fix_glitches: bool = False):
    glitched_count = 0

    for j in prange(precision.height):
        for i in range(precision.width):
            if fix_glitches and iterations_grid[j, i] != GLITCH_ITER:
                continue
            if not precision.approximate_pixel(i, j, iterations_grid):
                glitched_count += 1

    return glitched_count


@njit(fastmath=True, parallel=True, nogil=True)
def mandelbrot(b_left, t_right, height, width, iters):
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    b_left_r = b_left.real
    b_left_i = b_left.imag
    hor_step = (t_right.real - b_left_r)/width
    ver_step = (t_right.imag - b_left_i)/height

    for y in prange(height):
        c_imag = b_left_i + y * ver_step
        for x in prange(width):
            c_real = b_left_r + x * hor_step
            z_real = z_imag = 0
            for i in range(iters):
                temp = z_real
                z_real = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2 * temp * z_imag + c_imag

                if z_real * z_real + z_imag * z_imag > BREAKOUT_R_2:
                    iterations_grid[y, x] = i + 1
                    break
            if i == iters - 1:
                iterations_grid[y, x] = 0

    return iterations_grid


class Mandelbrot:
    def __init__(self, width: int, height: int, b_left: ComplexBf, t_right: ComplexBf, iterations: int,
                 multiprocessing: bool, cython: bool, pertubations: bool, num_series_terms, num_probes):
        self.w = width
        self.h = height
        self.corners_stack = []

        setcontext(context=Context(precision=200))

        self.init_corners = (b_left, t_right)
        self._set_corners(b_left=b_left, t_right=t_right)
        self.iterations = iterations
        self.cython = cython
        self.multiprocessing = multiprocessing

        self.pertubations = pertubations
        self.num_series_terms = num_series_terms
        self.num_probes = num_probes

        self.pixels: np.array = None

    def reset(self):
        self._set_corners(*self.init_corners)
        self.corners_stack = []

    def pop_corners(self):
        if not self.corners_stack:
            return
        self._set_corners(*self.corners_stack.pop())

    def reposition(self, b_left_coords: tuple, t_right_coords: tuple):
        b = self.b_left
        t = self.t_right

        hor_scale = (t.real - b.real)/self.w
        ver_scale = (t.imag - b.imag)/self.h

        b_left = ComplexBf(b.real + hor_scale*b_left_coords[0], b.imag + ver_scale*b_left_coords[1])
        t_right = ComplexBf(b.real + hor_scale*t_right_coords[0], b.imag + ver_scale*t_right_coords[1])

        if hasattr(self, "b_left"):
            self.corners_stack.append((self.b_left, self.t_right))
        self._set_corners(b_left, t_right)

    def _set_corners(self, b_left: ComplexBf, t_right: ComplexBf):
        height = float(t_right.imag - b_left.imag)
        width = float(t_right.real - b_left.real)

        ratio_target = self.h/self.w
        ratio_curr = height/width

        if ratio_target > ratio_curr:
            diff = BigFloat((width * ratio_target - height)/2)
            t_right.imag += diff
            b_left.imag -= diff
        else:
            diff = BigFloat((height / ratio_target - width) / 2)
            t_right += diff
            b_left -= diff

        self.b_left, self.t_right = b_left, t_right

    def getPixels(self):
        if self.pertubations:
            self.pixels = np.array(mandelbrot_pertubation(self.b_left, self.t_right, self.h, self.w, self.iterations, self.num_probes, self.num_series_terms), dtype=np.int32)
        elif not self.cython:
            self.pixels = mandelbrot(complex(self.b_left), complex(self.t_right), self.h, self.w, self.iterations)
        else:
            self.pixels = iterate(self.b_left, self.t_right, self.h, self.w, self.iterations, self.multiprocessing, self.pertubations)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)
