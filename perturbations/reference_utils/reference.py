from dataclasses import dataclass

import numpy as np
from gmpy2 import mpc, mpfr

from utils.constants import (
    BREAKOUT_R2,
)


@dataclass(eq=False)
class Reference:
    orbit: np.ndarray
    escaped_at: int
    init: mpc
    width: mpfr
    precision: int

    accurate_iters: int = 0
    scaling_factor: float = 1
    series_terms: np.ndarray = None

    def __hash__(self):
        return hash(
            (str(self.init), self.width, self.scaling_factor, self.accurate_iters)
        )

    def set_series_terms(
        self,
        series_terms: np.ndarray,
        scaling_factor: float,
        accurate_iters: np.ndarray,
    ):
        self.series_terms = series_terms
        self.scaling_factor = scaling_factor
        self.accurate_iters = accurate_iters


# TODO does this actually save time?
def iterate_ref(init_ref: mpc, iterations):
    ref_hist = np.zeros(iterations, dtype=np.complex128)
    ref = init_ref
    for i in range(iterations):
        temp = complex(ref)
        ref_hist[i] = temp
        if temp.real * temp.real + temp.imag * temp.imag > BREAKOUT_R2:
            return ref_hist, i + 1

        ref = ref * ref + init_ref

    return ref_hist, iterations
