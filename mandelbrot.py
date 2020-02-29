import math
import random

import numpy as np
from numba import jit
from bigfloat import BigFloat, Context, setcontext
from complex_bf import ComplexBf
from iterate import iterate

INIT_BOTTOM_LEFT = ComplexBf(BigFloat(-1.5), BigFloat(-1.25))
INIT_TOP_RIGHT = ComplexBf(BigFloat(0.5), BigFloat(1.25))
DEFAULT_ITER = 1000

ERROR_THRESH = 0.5
def compute_series_constants(b_left: ComplexBf, t_right: ComplexBf, seed_x_ratio, seed_y_ratio, iterations: int):
    a = np.zeros(iterations, dtype=np.complex_)
    b = np.zeros(iterations, dtype=np.complex_)
    c = np.zeros(iterations, dtype=np.complex_)

    a[0], b[0], c[0] = 1, 0, 0

    x = ComplexBf(seed_x_ratio*t_right.real - (1-seed_x_ratio)*b_left.real, seed_y_ratio*t_right.imag - (1-seed_y_ratio)*b_left.imag)
    z_hist = np.array(iterations, dtype=np.complex_)
    z_curr = x

    probes = [(b_left + x)/2, (t_right + x)/2]
    probe_deltas = [complex(p - x) for p in probes]
    probe_deltas_2 = [complex((p-x)*(p-x)) for p in probes]
    probe_deltas_3 = [complex((p-x)*(p-x)*(p-x)) for p in probes]
    z_del_actual = probe_deltas.copy()
    error_thresh_met = False
    z_escaped_at = 0
    for i in range(iterations):
        z_comp = complex(z_curr)
        z_hist[i] = z_comp
        a[i+1], b[i+1], c[i+1] = 2 * z_comp * a[i] + 1, 2 * z_comp * b[i] + a[i] * a[i], 2 * z_comp * c[i] + 2 * a[i] * b[i]
        z_curr = z_curr*z_curr + x
        if not z_escaped_at and z_curr.abs_2() > 2:
            z_escaped_at = i

        for j in range(len(probes)):
            del_n = z_del_actual[j]
            z_del_actual[j] = 2*z_comp*del_n + del_n*del_n + probe_deltas[j]

            z_del_app = a*probe_deltas[j] + b*probe_deltas_2[j] + c*probe_deltas_3[j]
            if abs(z_del_actual[j] - z_del_app) > ERROR_THRESH:
                error_thresh_met = True
                break

        if error_thresh_met:
            break

    return a, b, c, z_hist, z_curr, x, i, z_escaped_at


def iterate_pertubation(b_left: ComplexBf, t_right: ComplexBf, height, width, iterations):
    print("computing series constants...")
    a, b, c, z_hist, z, x, iter_accurate, z_escaped_at = compute_series_constants(b_left, t_right, 0.5, 0.5, iterations)
    # if z escaped too early (0 means z didn't escape)
    while not z_escaped_at and z_escaped_at < iterations*0.5:
        print("computing series constants...")
        a, b, c, z_hist, z, x, iter_accurate, z_escaped_at = compute_series_constants(b_left, t_right, random.random(),
                                                                                      random.random(), iterations)

    print("done, computing iterations")
    breakout = z_escaped_at
    for i in range(iter_accurate, iterations):
        if z.abs_2() > 4 and not breakout:
            breakout = i
        z_hist[i] = complex(z)
        z = z*z + x

    width_per_pixel = float(t_right.real - b_left.real)/width
    height_per_pixel = float(t_right.imag - b_left.imag)/height
    x_coords = (int((x.real - b_left.real)/width_per_pixel), int((x.imag - b_left.imag)/height_per_pixel))
    grid = _approximate_pixels(width_per_pixel, width, height_per_pixel, height, x_coords, a, b, c, breakout,
                               z_hist, iter_accurate)
    return grid


@jit(nopython=True)
def _approximate_pixels(width_per_pixel, width, height_per_pixel, height, x_coords, a, b, c, breakout,
                        precise_reference, iter_accurate):
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    deltas_iter = np.zeros((height, width), dtype=np.complex_)
    for i in range(width):
        for j in range(height):
            hor_delta = (x_coords[0] - i)*width_per_pixel
            ver_delta = (x_coords[1] - j)*height_per_pixel
            delta = math.sqrt(hor_delta*hor_delta + ver_delta*ver_delta)

            deltas_iter[j, i] = a[iter_accurate+1]*delta + b[iter_accurate+1]*delta*delta + c[iter_accurate+1]*delta*delta*delta

    deltas_iter_init = deltas_iter.copy()

    for x in range(width):
        for y in range(height):
            for i in range(breakout):
                delta_i = deltas_iter[y, x]
                if abs(delta_i + precise_reference[i]) < 2:
                    deltas_iter[y, x] = 2*precise_reference[i]*delta_i + delta_i*delta_i + delta_0
                else:
                    # if broke out before iterating, find true breakout value using binary search on accurate estimations

                    break
            if i == breakout - 1:
                iterations_grid[y, x] = 0
            else:
                iterations_grid[y, x] = i + iter_accurate

    return iterations_grid


@jit(nopython=True)
def mandelbrot(b_left, t_right, height, width, iters):
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    b_left_r = b_left.real
    b_left_i = b_left.imag
    hor_step = (t_right.real - b_left_r)/width
    ver_step = (t_right.imag - b_left_i)/height
    max_r_squared = 2**2

    for x in range(width):
        c_real = b_left_r + x * hor_step
        for y in range(height):
            c_imag = b_left_i + y * ver_step
            z_real = z_imag = 0
            for i in range(iters):
                temp = z_real
                z_real = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2 * temp * z_imag + c_imag

                if z_real * z_real + z_imag * z_imag > max_r_squared:
                    iterations_grid[y, x] = i + 1
                    break
            if i == iters - 1:
                iterations_grid[y, x] = 0

    return iterations_grid


class Mandelbrot:
    def __init__(self, width: int, height: int, b_left: ComplexBf, t_right: ComplexBf, iterations: int,
                 multiprocessing: bool, cython: bool, precise: bool):
        if None in {b_left.real, b_left.imag, t_right.real, t_right.imag}:
            b_left, t_right = INIT_BOTTOM_LEFT, INIT_TOP_RIGHT
        if iterations is None:
            iterations = DEFAULT_ITER

        self.w = width
        self.h = height
        self.corners_stack = []

        setcontext(context=Context(precision=200))

        self.init_corners = (b_left, t_right)
        self._set_corners(b_left=b_left, t_right=t_right)
        self.iterations = iterations
        self.cython = cython
        self.multiprocessing = multiprocessing
        self.precise = precise
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
        if self.precise:
            self.pixels = iterate_pertubation(self.b_left, self.t_right, self.h, self.w, self.iterations)
        elif not self.cython:
            self.pixels = mandelbrot(self.b_left, self.t_right, self.h, self.w, self.iterations)
        else:
            self.pixels = iterate(self.b_left, self.t_right, self.h, self.w, self.iterations, self.multiprocessing, self.precise)
        # elif self.multiprocessing and self.precise:
        #     self.pixels = iterate_multi_precise(self.b_left, self.t_right, self.h, self.w, self.iterations)
        # elif self.multiprocessing:
        #     self.pixels = iterate_multi(self.b_left, self.t_right, self.h, self.w, self.iterations)
        # else:
        #     self.pixels = iterate(self.b_left, self.t_right, self.h, self.w, self.iterations)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)
