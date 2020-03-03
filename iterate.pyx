import numpy as np
cimport openmp, cython
from cython.parallel cimport prange
from cython.parallel cimport parallel

BREAKOUT_R_2 = 10

def iterate(bottom_left, top_right, height, width, iterations, use_multiprocessing, high_precision):
    if use_multiprocessing:
        return multi(bottom_left.real, bottom_left.imag, top_right.real, top_right.imag, height, width, iterations)
    return base(bottom_left.real, bottom_left.imag, top_right.real, top_right.imag, height, width, iterations)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:,:] multi(double b_left_r, double b_left_i, double t_right_r, double t_right_i,
        int height, int width, int iterations):
    cdef int[:,:] iterations_grid = np.zeros((height, width), dtype=np.int)

    cdef double hor_step = (t_right_r - b_left_r)/width
    cdef double ver_step = (t_right_i - b_left_i)/height

    cdef double c_real, c_imag
    cdef double z_real, z_imag
    cdef double temp, max_r_squared = 4

    cdef int i, x, y, w = width, h = height, iters = iterations, max_r = BREAKOUT_R_2

    with nogil:
        for x in prange(w, schedule='static'):
            c_real = b_left_r + x*hor_step
            for y in prange(h, schedule='static'):
                c_imag = b_left_i + y*ver_step
                z_real = z_imag = 0
                for i in range(iters):
                    temp = z_real
                    z_real = z_real*z_real - z_imag*z_imag + c_real
                    z_imag = 2*temp*z_imag + c_imag

                    if z_real*z_real + z_imag*z_imag > max_r:
                        iterations_grid[y,x] = i + 1
                        break
                if i == iters - 1:
                    iterations_grid[y,x] = 0

    return iterations_grid


cdef int[:,:] base(long double b_left_r, long double b_left_i, long double t_right_r, long double t_right_i,
        int height, int width, int iterations):
    cdef int[:,:] iterations_grid = np.zeros((height, width), dtype=np.int)

    cdef double hor_step = (t_right_r - b_left_r)/width
    cdef double ver_step = (t_right_i - b_left_i)/height

    cdef double c_real, c_imag
    cdef double z_real, z_imag
    cdef double temp, max_r_squared = 4

    cdef int i, x, y, w = width, h = height, iters = iterations, max_r = BREAKOUT_R_2

    for x in range(w):
        c_real = b_left_r + x*hor_step
        for y in range(h):
            c_imag = b_left_i + y*ver_step
            z_real = z_imag = 0
            for i in range(iters):
                temp = z_real
                z_real = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2*temp*z_imag + c_imag

                if z_real*z_real + z_imag*z_imag > max_r:
                    iterations_grid[y,x] = i + 1
                    break
            if i == iters - 1:
                iterations_grid[y,x] = 0

    return iterations_grid
