from dataclasses import dataclass

from gmpy2 import mpc


@dataclass()
class MandelbrotConfig:
    width: int
    height: int
    t_left: mpc
    b_right: mpc
    iterations: int
    perturbation: bool
    gpu: bool
    num_series_terms: int