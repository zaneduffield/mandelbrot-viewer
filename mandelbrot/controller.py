from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from opencl.mandelbrot_cl import MandelbrotCL, ClassicMandelbrotCL
from perturbations.perturbation import PerturbationController
from utils.constants import BREAKOUT_R2
from utils.mandelbrot_utils import MandelbrotConfig
from .mandelbrot import mandelbrot


@dataclass()
class Node:
    config: MandelbrotConfig
    output: Tuple
    parent: "Node"
    children: List["Node"] = field(default_factory=list)


class MandelbrotController:
    def __init__(self):
        self.history: Node = None
        self._perturbations_controller: PerturbationController = None
        self._cl: MandelbrotCL = None

    def get_cl(self):
        if self._cl is None:
            self._cl = ClassicMandelbrotCL()
        return self._cl

    def get_pert(self):
        if self._perturbations_controller is None:
            self._perturbations_controller = PerturbationController()
        return self._perturbations_controller

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
            for output in self.get_pert().compute(config):
                yield output
        elif config.gpu:
            output = self.get_cl().compute(config)
        else:
            output = mandelbrot(config)

        if config.gpu:
            output = deepcopy(output)
        self.push(deepcopy(config), output)
        yield output


def convert_to_fractional_counts(iterations, points):
    abs_points = np.fmax(2, np.abs(points))
    return (
        iterations + np.log2(0.5 * np.log2(BREAKOUT_R2)) - np.log2(np.log2(abs_points))
    )
