import time

import numpy as np

from .perturbation_utils.opencl.mandelbrot_perturbation_cl import (
    PerturbationCL,
)
from .perturbation_utils.perturbation_state import make_perturbation_state
from .perturbation_utils.perturbed_iteration import iterate_perturbation
from .reference_utils.reference import Reference, iterate_ref
from .reference_utils.selection.blob_search import get_new_ref
from .reference_utils.selection.newton_iteration import locate_zero
from .reference_utils.series_approximation import compute_series_terms
from brot.utils.constants import (
    GLITCH_ITER,
    MAX_GLITCH_FIX_LOOPS,
    MAX_OK_GLITCH_COUNT,
)
from brot.utils.mandelbrot_utils import MandelbrotConfig, get_precision, my_logger


class PerturbationController:
    def __init__(self):
        self._cl = None

    def _get_cl(self):
        if self._cl is None:
            self._cl = PerturbationCL()
        return self._cl

    def _get_or_set_zero_arr(self, name, dtype: type, shape):
        if not hasattr(self, name) or self.__getattribute__(name).shape != shape:
            self.__setattr__(name, np.empty(shape, dtype=dtype))
        self.__getattribute__(name).fill(0)
        return self.__getattribute__(name)

    def _get_iterations_grid(self, config: MandelbrotConfig):
        shape = (config.image_height, config.image_width)
        return self._get_or_set_zero_arr("_iterations_grid", np.int32, shape)

    def _get_points_grid(self, config: MandelbrotConfig):
        shape = (config.image_height, config.image_width)
        return self._get_or_set_zero_arr("_points_grid", np.complex128, shape)

    def _get_ref(self, config: MandelbrotConfig, coords):
        reference = config.get_point_by_coords(*coords)
        start = time.time()
        my_logger.debug("iterating reference")
        ref_orbit, ref_escaped_at = iterate_ref(reference, config.max_iterations)
        my_logger.debug(f"iterating reference took {time.time() - start} seconds")

        my_logger.debug("computing series constants...")
        start = time.time()
        reference = Reference(
            orbit=ref_orbit,
            init=reference,
            escaped_at=ref_escaped_at,
            width=config.get_width(),
            precision=get_precision(),
        )
        compute_series_terms(config, reference)
        my_logger.debug(f"series constants took {time.time() - start} seconds")

        my_logger.debug(
            f"proceeding with {reference.accurate_iters} reference iterations"
        )
        my_logger.debug(f"reference broke out at {ref_escaped_at}")
        return reference

    def compute(self, config: MandelbrotConfig):
        config.set_precision()

        start = time.time()
        approx_zero = locate_zero(config)
        my_logger.debug(
            f"spent {time.time() - start:4f} seconds looking for an approx zero"
        )
        if approx_zero is not None and config.is_point_in_frame(approx_zero):
            my_logger.debug("approx zero found within frame!")
            ref_coords = config.get_coords_for_point(approx_zero)
        else:
            ref_coords = config.get_coords_for_point(config.get_center())

        loops = 0
        while loops <= MAX_GLITCH_FIX_LOOPS:
            reference = self._get_ref(config, ref_coords)
            perturbation_state = make_perturbation_state(
                config, np.array(ref_coords, dtype=np.int32), reference
            )
            my_logger.debug(f"new ref iterated at: {perturbation_state.ref_coords}")

            if config.gpu:
                iterations_grid, points = self._get_cl().compute(
                    perturbation_state,
                    fix_glitches=bool(loops),
                    double_precision=config.gpu_double_precision,
                )
            else:
                iterations_grid, points = (
                    self._get_iterations_grid(config),
                    self._get_points_grid(config),
                )
                iterate_perturbation(
                    iterations_grid,
                    points,
                    perturbation_state,
                    fix_glitches=bool(loops),
                )
            yield iterations_grid, points

            glitched_count = np.sum(iterations_grid == GLITCH_ITER)
            my_logger.debug(f"{glitched_count} pixels remaining")
            if glitched_count <= MAX_OK_GLITCH_COUNT:
                break

            ref_coords = get_new_ref(iterations_grid == GLITCH_ITER)
            my_logger.debug(f"new ref at :{ref_coords}")
            loops += 1
