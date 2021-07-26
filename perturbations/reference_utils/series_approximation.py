import math
import numpy as np

from gmpy2 import mpc
from numba import njit

from perturbations.reference_utils.reference import Reference
from utils.mandelbrot_utils import MandelbrotConfig


@njit
def iterate_series_terms(
    ref_hist,
    ref_escaped_at: int,
    probe_deltas,
    terms,
    num_terms: int,
    scaling_factor: float,
):
    """
    The error tolerance has a huge impact on the correctness of the series approximation
    Pretty much no matter how hard we try this won't be good enough in all cases and glitches
    will slip through. There are more advanced techniques for detecting when the series approximation is no longer
    accurate but I don't understand them yet.
    """
    tolerance = 1 / np.power(2, 48)
    num_terms = len(terms)

    scaled_delta_powers = np.zeros((len(probe_deltas), num_terms), dtype=np.complex128)
    for j in range(len(probe_deltas)):
        scaled_delta_powers[j, 0] = probe_deltas[j] * scaling_factor
        for k in range(1, num_terms):
            scaled_delta_powers[j, k] = (
                scaled_delta_powers[j, k - 1] * scaled_delta_powers[j, 0]
            )

    # This acts to reduce the size of our series terms and is reversed when computing using them
    terms[0][0] = 1 / scaling_factor
    for i in range(1, num_terms):
        terms[i][0] = 0

    for i in range(ref_escaped_at):
        z_comp = ref_hist[i]
        terms[0][i + 1] = 2 * z_comp * terms[0][i] + terms[0][0]
        for j in range(1, num_terms):
            s = 0
            for k in range(j):
                s += terms[k][i] * terms[j - k - 1][i]

            new_term = 2 * z_comp * terms[j][i] + s
            if not np.isnan(new_term):
                terms[j][i + 1] = new_term

        for j in range(len(scaled_delta_powers)):
            last = 1 / tolerance
            for k in range(num_terms):
                term = np.abs(terms[k][i + 1] * scaled_delta_powers[j][k])
                if last * tolerance < term:
                    return i
                last = term

    return ref_escaped_at


def get_probe_deltas(t_left: mpc, b_right: mpc, ref_init: mpc, num_probes):
    probes = []
    square_side_len = int(math.sqrt(num_probes))
    for i in range(square_side_len):
        for j in range(square_side_len):
            x_ratio, y_ratio = i / square_side_len, j / square_side_len
            probe_del = (
                x_ratio * (b_right.real - t_left.real)
                - y_ratio * (t_left.imag - b_right.imag) * 1j
            )
            probes.append(complex(t_left + probe_del - ref_init))

    return np.array(probes)


def compute_series_terms(
    config: MandelbrotConfig,
    ref: Reference,
    num_terms=5,
    num_probes=25,
):
    probe_deltas = get_probe_deltas(
        config.t_left(), config.b_right(), ref.init, num_probes
    )
    terms = np.zeros((num_terms, config.max_iterations), dtype=np.complex_, order="F")

    scaling_factor = float(1 / (config.b_right().real - config.t_left().real))
    accurate_iters = iterate_series_terms(
        ref.orbit, ref.escaped_at, probe_deltas, terms, num_terms, scaling_factor
    )
    ref.set_series_terms(terms, scaling_factor, accurate_iters)
