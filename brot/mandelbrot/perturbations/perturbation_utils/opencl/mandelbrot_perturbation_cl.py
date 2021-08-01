from pathlib import Path

import numpy as np

from brot.mandelbrot.opencl.mandelbrot_cl import MandelbrotCL, cl
from brot.mandelbrot.perturbations.perturbation_utils.perturbed_iteration import (
    PerturbationState,
)
from brot.utils.constants import (
    GLITCH_ITER,
    BREAKOUT_R2,
    GLITCH_DIFF_THRESHOLD,
)
from brot.utils.mandelbrot_utils import my_logger


class PerturbationCL(MandelbrotCL):
    def get_program_contents(self):
        with open(Path(__file__).parent / "mandelbrot_perturbation.cl") as f:
            return f.read()

    def _compute(self, pert: PerturbationState, fix_glitches):
        my_logger.debug("computing")
        real_dtype = self._get_real_dtype()
        complex_dtype = self._get_complex_dtype()

        terms_buf = cl.Buffer(
            self.ctx,
            self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
            hostbuf=pert.terms.astype(complex_dtype),
        )
        precise_ref_buf = cl.Buffer(
            self.ctx,
            self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
            hostbuf=pert.precise_reference.astype(complex_dtype),
        )

        self.prg.approximate_pixels(
            self.queue,
            self.iterations_grid.shape,
            None,
            self.ibuf,
            self.pbuf,
            real_dtype(pert.w_per_pix),
            real_dtype(pert.h_per_pix),
            np.int32(pert.width),
            np.int32(pert.ref_coords[1]),
            np.int32(pert.ref_coords[0]),
            terms_buf,
            np.int32(pert.num_terms),
            real_dtype(pert.scaling_factor),
            np.int32(pert.breakout),
            precise_ref_buf,
            np.int32(pert.iter_accurate),
            np.int32(pert.max_iterations),
            np.int32(GLITCH_ITER),
            real_dtype(GLITCH_DIFF_THRESHOLD),
            np.int32(fix_glitches),
            np.int32(BREAKOUT_R2),
        )

    def compute(
        self, pert: PerturbationState, fix_glitches: bool, double_precision: bool
    ):
        self.set_precision(double_precision)
        with self.manage_buffer(pert.height, pert.width):
            self._compute(pert, fix_glitches)
        return self.out
