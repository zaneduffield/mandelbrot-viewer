import math
import random
from typing import List

import numpy as np
from numba import jit, jitclass, njit, prange
from numba import int32, float32, float64, complex128, int64
from bigfloat import BigFloat, Context, setcontext
from complex_bf import ComplexBf
from iterate import iterate, BREAKOUT_R_2
from pertubations import mandelbrot_pertubation, get_new_ref


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
        self.best_ref = None
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
        height = t_right.imag - b_left.imag
        width = t_right.real - b_left.real

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
            self.pixels, self.best_ref = mandelbrot_pertubation(self.b_left, self.t_right, self.h, self.w, self.iterations, self.num_probes, self.num_series_terms, self.best_ref)
            self.pixels = np.array(self.pixels, dtype=np.int32)
        elif not self.cython:
            self.pixels = mandelbrot(complex(self.b_left), complex(self.t_right), self.h, self.w, self.iterations)
        else:
            self.pixels = iterate(self.b_left, self.t_right, self.h, self.w, self.iterations, self.multiprocessing, self.pertubations)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)
