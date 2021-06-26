from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

import numpy as np
from numba import njit, prange

from opencl.mandelbrot_cl import MandelbrotCL, ClassicMandelbrotCL
from perturbations.perturbations import PerturbationComputer
from utils.constants import BREAKOUT_R2, NUM_PROBES, NUM_SERIES_TERMS
from utils.mandelbrot_utils import MandelbrotConfig


@njit(fastmath=True, parallel=True, nogil=True)
def mandelbrot(t_left, b_right, height, width, max_iter):
    iterations_grid = np.zeros((height, width), dtype=np.int32)
    final_points = np.zeros((height, width), dtype=np.complex128)
    t_left_r = t_left.real
    t_left_i = t_left.imag
    hor_step = (b_right.real - t_left_r) / width
    ver_step = (t_left.imag - b_right.imag) / height

    for y in prange(height):
        c_imag = t_left_i - y * ver_step
        for x in prange(width):
            c_real = t_left_r + x * hor_step
            z_real = z_imag = 0

            i = 0
            while i < max_iter and z_real * z_real + z_imag * z_imag < BREAKOUT_R2:
                i += 1
                temp = z_real
                z_real = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2 * temp * z_imag + c_imag

            iterations_grid[y, x] = i
            final_points[y, x] = z_real + 1j * z_imag

    return iterations_grid, final_points


@dataclass()
class Node:
    config: MandelbrotConfig
    iteration_grid: np.array
    final_points: np.array
    parent: 'Node'
    children: List['Node'] = field(default_factory=list)


class Mandelbrot:
    def __init__(self):
        self.history: Node = None
        self._perturbations_computer: PerturbationComputer = None
        self._cl: MandelbrotCL = None

    def get_cl(self):
        if self._cl is None:
            self._cl = ClassicMandelbrotCL()
        return self._cl

    def get_pert(self):
        if self._perturbations_computer is None:
            self._perturbations_computer = PerturbationComputer()
        return self._perturbations_computer

    def push(self, config: MandelbrotConfig, iteration_grid: np.array, final_points: np.array):
        node = Node(config, iteration_grid, final_points, self.history)
        if self.history is not None:
            self.history.children.append(node)
        self.history = node

    def back(self):
        if self.history is None:
            return
        out = self.history.parent
        if out is not None:
            self.history = out
            return deepcopy(out.config), out.iteration_grid, out.final_points

    def next(self):
        if self.history.children:
            self.history = self.history.children[-1]
            return deepcopy(self.history.config), self.history.iteration_grid

    def compute(self, config: MandelbrotConfig):
        iteration_grid, final_points = None, None
        if config.perturbation:
            for iteration_grid, final_points in self.get_pert().compute(config, NUM_PROBES, NUM_SERIES_TERMS):
                yield iteration_grid, final_points
        elif config.gpu:
            iteration_grid, final_points = self.get_cl().compute(config)
        else:
            iteration_grid, final_points = mandelbrot(complex(config.t_left), complex(config.b_right), config.height,
                                                      config.width,
                                                      config.iterations)
        if config.gpu:
            iteration_grid, final_points = deepcopy(iteration_grid), deepcopy(final_points)

        self.push(deepcopy(config), iteration_grid, final_points)
        yield iteration_grid, final_points
