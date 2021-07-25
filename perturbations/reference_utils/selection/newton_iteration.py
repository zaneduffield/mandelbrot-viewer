import time

from gmpy2 import mpc

from utils.constants import BREAKOUT_R2
from utils.mandelbrot_utils import MandelbrotConfig, my_logger


def locate_zero(config: MandelbrotConfig):
    start = time.time()
    period = find_period(config)
    if period is None:
        my_logger.debug(
            f"Couldn't detect point of at most period {config.max_iterations}"
        )
        return

    my_logger.debug(f"Detected periodic point in {time.time() - start} seconds")
    start = time.time()

    def outside_real_range(x):
        return outside_range(x, config.t_left().real, config.b_right().real)

    def outside_imag_range(x):
        return outside_range(x, config.b_right().imag, config.t_left().imag)

    c = config.get_center()
    threshold = config.get_width_per_pix() ** 2
    loops = 0
    while True:
        z, dzdc, escaped_at = iterate_with_derivative(c, period)
        if z.real * z.real + z.imag * z.imag < threshold:
            my_logger.debug(
                f"found point with period {period} in {loops} cycles of Newton iteration in {time.time() - start} seconds"
            )
            return c
        elif escaped_at < period:
            my_logger.error(f"iterated point escaped earlier than the period found")
            return

        loops += 1
        c -= z / dzdc

        if outside_real_range(c.real) or outside_imag_range(c.imag):
            my_logger.debug("point was iterated outside the current view, aborting")
            return


def outside_range(x, lower, upper):
    return x < lower or x > upper


def iterate_with_derivative(c: mpc, iterations):
    z = 0
    dzdc = 0
    i = 0
    while i < iterations:
        i += 1
        dzdc = 2 * z * dzdc + 1
        z = z * z + c

        if z.real * z.real + z.imag * z.imag > BREAKOUT_R2:
            break

    return z, dzdc, i


def find_period(config: MandelbrotConfig):
    """
    The idea is to iterate a polygon containing region in which we would like to find a periodic point.
    A periodic point of period n is a root of the mandelbrot iteration of degree n.
    We iterate the polygon until it encloses the origin. When this happens, it's very likely that there 
    exists a point from the interior of the original polygon that is a root at this iteration number. 
    This is because the iterated mandelbrot iteration formula is continuous and the original polygon is 
    very small.
    Return this number as the target period.
    """
    c_j = (
        config.t_left(),
        config.t_left() + config.get_width(),
        config.b_right(),
        config.b_right() - config.get_width(),
    )
    z_i = [0] * len(c_j)
    for i in range(config.max_iterations):
        for j in range(len(c_j)):
            z_i[j] = z_i[j] * z_i[j] + c_j[j]
            # technically these points could escape before a period is detected but it's so unlikely
            # (and unimportant when it occurs) that the cost of checking is not worth it

        if surrounds_origin(z_i):
            return i + 1


def surrounds_origin(vertices):
    # If an odd number of the edges of a polygon cross the positive real axis,
    # then the origin is contained within the polygon.
    odd_crosses = False
    for j in range(len(vertices)):
        if crosses_positive_real_axis(vertices[j - 1], vertices[j]):
            odd_crosses = not odd_crosses
    return odd_crosses


def crosses_positive_real_axis(a: mpc, b: mpc):
    if (a.imag > 0) != (b.imag > 0):
        diff = b - a
        # The equation of the line between a and b is
        #   y = (diff.imag / diff.real) * (x - b.real) + b.imag
        # Intersect this with the line y = 0 and solve for x > 0
        return 0 < b.real - b.imag * (diff.real / diff.imag)
    return False
