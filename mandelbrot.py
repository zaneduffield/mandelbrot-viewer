import math
import random
from typing import List

import numpy as np
from numba import jit, jitclass, njit, prange
from numba import int32, float32, float64, complex128, int64
from bigfloat import BigFloat, Context, setcontext
from complex_bf import ComplexBf
from opencl_test import MandelbrotCL

BREAKOUT_R_2 = 20
from pertubations import mandelbrot_pertubation, get_new_ref


@njit(fastmath=True, parallel=True, nogil=True)
def mandelbrot(t_left, b_right, height, width, iters):
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    t_left_r = t_left.real
    t_left_i = t_left.imag
    hor_step = (b_right.real - t_left_r)/width
    ver_step = (t_left.imag - b_right.imag)/height

    for y in prange(height):
        c_imag = t_left_i - y * ver_step
        for x in prange(width):
            c_real = t_left_r + x * hor_step
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
    def __init__(self, width: int, height: int, t_left: ComplexBf, b_right: ComplexBf, iterations: int,
                 multiprocessing: bool, gpu: bool, pertubations: bool, num_series_terms, num_probes):
        self.w = width
        self.h = height
        self.corners_stack = []

        self.init_corners = (t_left, b_right)
        self._set_corners(t_left=t_left, b_right=b_right)
        self.iterations = iterations
        self.multiprocessing = multiprocessing

        self.pertubations = pertubations
        self.num_series_terms = num_series_terms
        self.num_probes = num_probes

        self.gpu = gpu
        self.cl = None
        self.set_gpu(gpu)

        self.pixels: np.array = None

    def reset(self):
        self._set_corners(*self.init_corners)
        self.corners_stack = []

    def set_gpu(self, gpu: bool):
        self.gpu = gpu
        if self.gpu and self.cl is None:
            self.cl = MandelbrotCL(self.w, self.h)

    def pop_corners(self):
        if not self.corners_stack:
            return
        self._set_corners(*self.corners_stack.pop())

    def reposition(self, t_left_coords: tuple, b_right_coords: tuple):
        b = self.b_right
        t = self.t_left

        hor_scale = (b.real - t.real)/self.w
        ver_scale = (t.imag - b.imag)/self.h

        t_left = ComplexBf(t.real + hor_scale*t_left_coords[0], t.imag - ver_scale*t_left_coords[1])
        b_right = ComplexBf(t.real + hor_scale*b_right_coords[0], t.imag - ver_scale*b_right_coords[1])

        self.corners_stack.append((self.t_left, self.b_right))
        self._set_corners(t_left, b_right)

    def _set_corners(self, t_left: ComplexBf, b_right: ComplexBf):
        height = t_left.imag - b_right.imag
        width = b_right.real - t_left.real
        setcontext(context=Context(precision=200))

        ratio_target = self.h/self.w
        ratio_curr = height/width

        if ratio_target > ratio_curr:
            diff = BigFloat((width * ratio_target - height)/2)
            t_left.imag += diff
            b_right.imag -= diff
        else:
            diff = BigFloat((height / ratio_target - width) / 2)
            t_left -= diff
            b_right += diff

        self.t_left, self.b_right = t_left, b_right

    def get_width(self):
        return float(self.b_right.real - self.t_left.real)

    def get_pixels(self):
        if self.pertubations:
            self.pixels = mandelbrot_pertubation(self.t_left, self.b_right, self.h, self.w, self.iterations, self.num_probes, self.num_series_terms)
            self.pixels = np.array(self.pixels, dtype=np.int32)
        elif self.gpu:
            self.pixels = self.cl.get_pixels(self.t_left, self.b_right, self.h, self.w, self.iterations)
        else:
            self.pixels = mandelbrot(complex(self.t_left), complex(self.b_right), self.h, self.w, self.iterations)



def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)
