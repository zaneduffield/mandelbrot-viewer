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
        self._zoom = zoom
        self.set_precision()
        self._compute_corners()

    def set_center(self, center: mpfr):
        self._center = center
        self._compute_corners()

    def set_precision(self):
        set_precision(self.get_width() / max(1000, self.image_width))

    def get_center(self):
        return self._center

    def _convert_between_zoom_width(self, value):
        return 4 / value

    def get_zoom(self):
        return self._zoom

    def get_width(self):
        return self._convert_between_zoom_width(self.get_zoom())

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


def set_precision(width_per_pixel: mpfr):
    get_context().precision = max(10, int(-log2(width_per_pixel) * 2))
