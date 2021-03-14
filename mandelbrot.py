from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

import numpy as np
from numba import njit, prange

from constants import BREAKOUT_R_2, NUM_PROBES, NUM_SERIES_TERMS
from mandelbrot_utils import MandelbrotConfig
from opencl_test import MandelbrotCL, ClassicMandelbrotCL
from perturbations import PerturbationComputer


@njit(fastmath=True, parallel=True, nogil=True)
def mandelbrot(t_left, b_right, height, width, iters):
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    t_left_r = t_left.real
    t_left_i = t_left.imag
    hor_step = (b_right.real - t_left_r) / width
    ver_step = (t_left.imag - b_right.imag) / height

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


@dataclass()
class Node:
    config: MandelbrotConfig
    pixels: np.array
    parent: 'Node'
    children: List['Node'] = field(default_factory=list)


class Mandelbrot:
    def __init__(self):
        self.history: Node = None
        self.init_corners = None
        self._perturbations_computer: PerturbationComputer = None
        self._cl: MandelbrotCL = None
        self.pixels: np.array = None

    def get_cl(self):
        if self._cl is None:
            self._cl = ClassicMandelbrotCL()
        return self._cl

    def get_pert(self):
        if self._perturbations_computer is None:
            self._perturbations_computer = PerturbationComputer()
        return self._perturbations_computer

    def push(self, config: MandelbrotConfig, pixels: np.array):
        node = Node(deepcopy(config), pixels, self.history)
        if self.history is not None:
            self.history.children.append(node)
        self.history = node

    def back(self):
        if self.history is None:
            return
        out = self.history.parent
        if out is not None:
            self.history = out
            return deepcopy(out.config), out.pixels

    def next(self):
        if self.history.children:
            self.history = self.history.children[-1]
            return self.history.config, self.history.pixels

    def get_pixels(self, config: MandelbrotConfig):
        if config.perturbation:
            pixels = None
            for iterative_pixels in self.get_pert().compute(config, NUM_PROBES, NUM_SERIES_TERMS):
                pixels = iterative_pixels
                yield pixels
            self.push(config, pixels)
        elif config.gpu:
            pixels = self.get_cl().get_pixels(config)
            self.push(config, pixels)
            yield pixels
        else:
            pixels = mandelbrot(complex(config.t_left), complex(config.b_right), config.height, config.width,
                                config.iterations)
            self.push(config, pixels)
            yield pixels
