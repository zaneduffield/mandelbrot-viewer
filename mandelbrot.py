import numpy as np
from numba import njit, prange, set_num_threads
from multiprocessing import cpu_count
from gmpy2 import mpc, get_context

from constants import BREAKOUT_R_2
from opencl_test import MandelbrotCL, ClassicMandelbrotCL
from perturbations import PerturbationComputer


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
    def __init__(self, width: int, height: int, t_left: mpc, b_right: mpc, iterations: int,
                 multiprocessing: bool, gpu: bool, perturbations: bool, num_series_terms, num_probes):
        self.w = width
        self.h = height
        self.corners_stack = []

        self.init_corners = (t_left, b_right)
        self._set_corners(t_left=t_left, b_right=b_right)
        self.iterations = iterations
        self.multiprocessing = multiprocessing
        self.set_max_threads()

        self._perturbations = perturbations
        self._perturbations_computer: PerturbationComputer = None
        self.num_series_terms = num_series_terms
        self.num_probes = num_probes

        self._gpu = gpu
        self._cl: MandelbrotCL = None
        self.set_gpu(gpu)

        self.pixels: np.array = None

    def reset(self):
        self._set_corners(*self.init_corners)
        self.corners_stack = []

    def set_gpu(self, gpu: bool):
        self._gpu = gpu

    def load_cl(self):
        if self._cl is None:
            self._cl = ClassicMandelbrotCL()

    def set_perturbations(self, perturbations: bool):
        self._perturbations = perturbations

    def load_pert(self):
        if self._perturbations_computer is None:
            self._perturbations_computer = PerturbationComputer()

    def pop_corners(self):
        if not self.corners_stack:
            return
        self._set_corners(*self.corners_stack.pop())

    def reposition(self, t_left_coords: tuple, b_right_coords: tuple):
        b = self.b_right
        t = self.t_left

        hor_scale = (b.real - t.real)/self.w
        ver_scale = (t.imag - b.imag)/self.h

        t_left = t + mpc(hor_scale*t_left_coords[0] - ver_scale*t_left_coords[1]*1j)
        b_right = t + mpc(hor_scale*b_right_coords[0] - ver_scale*b_right_coords[1]*1j)

        self.corners_stack.append((self.t_left, self.b_right))
        self._set_corners(t_left, b_right)

    def _set_corners(self, t_left: mpc, b_right: mpc):
        height = t_left.imag - b_right.imag
        width = b_right.real - t_left.real

        get_context().precision = int(-np.log2(float(width/self.w)))

        ratio_target = self.h/self.w
        ratio_curr = height/width

        if ratio_target > ratio_curr:
            diff = (width * ratio_target - height)/2
            t_left += diff*1j
            b_right -= diff*1j
        else:
            diff = (height / ratio_target - width) / 2
            t_left -= diff
            b_right += diff

        self.t_left, self.b_right = t_left, b_right

    def get_width(self):
        return float(self.b_right.real - self.t_left.real)

    def set_max_threads(self):
        set_num_threads(cpu_count() - 1 if self.multiprocessing else 1)

    def get_pixels(self):
        self.set_max_threads()
        if self._perturbations:
            self.load_pert()
            yield from self._perturbations_computer.compute(self.t_left, self.b_right, self.h, self.w, self.iterations, self.num_probes, self.num_series_terms, self._gpu)
        elif self._gpu:
            self.load_cl()
            yield self._cl.get_pixels(self.t_left, self.b_right, self.h, self.w, self.iterations)
        else:
            yield mandelbrot(complex(self.t_left), complex(self.b_right), self.h, self.w, self.iterations)
