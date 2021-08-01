import numpy as np
from numba import njit, prange

from utils.constants import BREAKOUT_R2
from utils.mandelbrot_utils import MandelbrotConfig


def mandelbrot(config: MandelbrotConfig):
    return _mandelbrot(
        complex(config.t_left()),
        complex(config.b_right()),
        config.image_height,
        config.image_width,
        config.max_iterations,
    )


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
