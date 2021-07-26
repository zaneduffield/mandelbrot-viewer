from numba import float64, complex128, int32
from numba.experimental import jitclass

from perturbations.reference_utils.reference import Reference
from utils.mandelbrot_utils import MandelbrotConfig


def make_perturbation_state(config: MandelbrotConfig, ref_coords, reference: Reference):
    return PerturbationState(
        config.get_width_per_pix(),
        config.get_height_per_pix(),
        config.image_width,
        config.image_height,
        ref_coords,
        reference.series_terms,
        reference.series_terms.shape[0],
        reference.escaped_at,
        reference.orbit,
        reference.accurate_iters,
        config.max_iterations,
        reference.scaling_factor,
    )


@jitclass(
    [
        ("w_per_pix", float64),
        ("h_per_pix", float64),
        ("width", int32),
        ("height", int32),
        ("ref_coords", int32[:]),
        ("terms", complex128[:, :]),
        ("num_terms", int32),
        ("breakout", int32),
        ("precise_reference", complex128[:]),
        ("iter_accurate", int32),
        ("max_iterations", int32),
        ("scaling_factor", float64),
    ]
)
class PerturbationState:
    def __init__(
        self,
        w_per_pix,
        h_per_pix,
        width,
        height,
        ref_coords,
        terms,
        num_terms,
        breakout,
        precise_reference,
        iter_accurate,
        max_iterations,
        scaling_factor,
    ):
        self.w_per_pix = w_per_pix
        self.h_per_pix = h_per_pix
        self.width = width
        self.height = height
        self.ref_coords = ref_coords
        self.terms, self.num_terms = terms, num_terms
        self.breakout = breakout
        self.precise_reference = precise_reference
        self.iter_accurate = iter_accurate
        self.max_iterations = max_iterations
        self.scaling_factor = scaling_factor
