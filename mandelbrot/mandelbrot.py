from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numba import njit, prange

from opencl.mandelbrot_cl import MandelbrotCL, ClassicMandelbrotCL
from perturbations.perturbations import PerturbationComputer
from utils.constants import BREAKOUT_R2, NUM_PROBES, NUM_SERIES_TERMS
from utils.mandelbrot_utils import MandelbrotConfig


def mandelbrot(config: MandelbrotConfig):
    return _mandelbrot(complex(config.t_left()), complex(config.b_right()), config.image_height, config.image_width, config.max_iterations)


@njit(fastmath=True, parallel=True, nogil=True)
def _mandelbrot(t_left, b_right, height, width, max_iter):
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
    output: Tuple
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

    def push(self, config: MandelbrotConfig, output):
        node = Node(config, output, self.history)
        if self.history is not None:
            self.history.children.append(node)
        self.history = node

    def back(self):
        if self.history is None:
            return
        prev = self.history.parent
        if prev is not None:
            self.history = prev
            return deepcopy(prev.config), prev.output

    def next(self):
        if self.history.children:
            self.history = self.history.children[-1]
            return deepcopy(self.history.config), self.history.output

    def compute(self, config: MandelbrotConfig):
        output = (None, None)
        if config.perturbation:
            for output in self.get_pert().compute(config, NUM_PROBES, NUM_SERIES_TERMS):
                yield output
        elif config.gpu:
            output = self.get_cl().compute(config)
        else:
            output = mandelbrot(config)

        if config.gpu:
            output = deepcopy(output)
        self.push(deepcopy(config), output)
        yield output


def convert_to_fractional_counts(iterations, points, scale=100):
    abs_points = np.maximum(2, np.abs(points))
    return scale * (iterations + np.log2(0.5*np.log2(BREAKOUT_R2)) - np.log2(np.log2(abs_points)))

