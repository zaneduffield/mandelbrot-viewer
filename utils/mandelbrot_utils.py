import logging
from dataclasses import dataclass

from gmpy2 import mpc, mpfr, get_context, log2

logging.basicConfig(format="%(levelname)s: %(message)s")
my_logger = logging.getLogger(__name__)


@dataclass()
class MandelbrotConfig:
    image_width: int
    image_height: int
    max_iterations: int
    perturbation: bool
    gpu: bool
    gpu_double_precision: bool = False

    _center: mpc = None
    _zoom: mpfr = None
    _t_left: mpc = None
    _b_right: mpc = None

    def set_center_and_zoom(self, center: mpc, zoom: mpfr):
        self._center = center
        self.set_zoom(zoom)

    def set_image_dimensions(self, width: int, height: int):
        self.image_width = width
        self.image_height = height
        self._compute_corners()

    def set_zoom(self, zoom: mpfr):
        if zoom <= 0:
            raise ValueError("zoom must be positive")
        self._zoom = zoom
        self.set_precision()
        self._compute_corners()

    def set_center(self, center: mpfr):
        self._center = center
        self._compute_corners()

    def get_required_precision(self):
        return required_precision_for_pix_width(
            self.get_width() / max(1000, self.image_width)
        )

    def set_precision(self, extra_precision_factor=1):
        set_precision(self.get_required_precision() * extra_precision_factor)

    def get_center(self):
        return self._center

    def _convert_between_zoom_width(self, value):
        return 4 / value

    def get_zoom(self):
        return self._zoom

    def get_width(self):
        return self._convert_between_zoom_width(self.get_zoom())

    def get_width_per_pix(self):
        return float((self.b_right().real - self.t_left().real) / self.image_width)

    def get_height_per_pix(self):
        return float((-self.b_right().imag + self.t_left().imag) / self.image_height)

    def get_point_by_coords(self, imag: int, real: int):
        return (
            self.t_left()
            + real * self.get_width_per_pix()
            - imag * self.get_height_per_pix() * 1j
        )

    def get_coords_for_point(self, point: mpc):
        return (
            (point.real - self.t_left().real) / self.get_width_per_pix(),
            (-point.imag + self.t_left().imag) / self.get_height_per_pix(),
        )

    def set_zoom_from_width(self, width: mpfr):
        self.set_zoom(self._convert_between_zoom_width(width))

    def _compute_corners(self):
        half_width = self.get_width() / 2
        half_height = half_width * self.image_height / self.image_width
        half_diag = 1j * half_height - half_width
        self._t_left = self.get_center() + half_diag
        self._b_right = self.get_center() - half_diag

    def t_left(self):
        if self._t_left is None:
            self._compute_corners()
        return self._t_left

    def b_right(self):
        if self._b_right is None:
            self._compute_corners()
        return self._b_right


def required_precision_for_pix_width(width_per_pixel):
    return max(10, int(-log2(width_per_pixel) * 2))


def set_precision(precision: int):
    get_context().precision = precision


def get_precision():
    return get_context().precision
