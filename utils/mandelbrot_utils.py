import logging
from dataclasses import dataclass

from gmpy2 import mpc, mpfr, get_context, log2

logging.basicConfig(format="%(levelname)s: %(message)s")
my_logger = logging.getLogger(__name__)


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


def set_precision(width_per_pixel: mpfr):
    get_context().precision = int(-log2(width_per_pixel) * 2)


def set_precision_from_config(config: MandelbrotConfig):
    width = config.b_right.real - config.t_left.real
    pix_width = config.width
    return set_precision(width / pix_width)
