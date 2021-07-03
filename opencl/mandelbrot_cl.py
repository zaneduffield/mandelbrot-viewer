from pathlib import Path

import numpy as np
# import os
# os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
from gmpy2 import mpc

from utils.constants import BREAKOUT_R2
from utils.mandelbrot_utils import MandelbrotConfig, my_logger

PY_OPEN_CL_INSTALLED = False
cl = None
try:
    import pyopencl as cl
    PY_OPEN_CL_INSTALLED = True
except ModuleNotFoundError:
    my_logger.warn("No PyOpenCL installation found.")


class MandelbrotCL:
    def __init__(self):
        platform = cl.get_platforms()[0]
        my_logger.debug(platform.get_devices())

        self.mf = cl.mem_flags
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.shape = None
        self.prg = None
        self.out = None

    def compile(self, program_contents: str):
        my_logger.debug("compiling...")
        self.prg = cl.Program(self.ctx, program_contents).build()
        my_logger.debug("done")

    def set_arrays(self, height, width):
        if self.shape != (height, width):
            self.shape = (height, width)
            self.iterations_grid = np.zeros(self.shape, dtype=np.int32, order="C")
            self.ibuf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.iterations_grid.nbytes)
            self.points_grid = np.zeros(self.shape, dtype=np.complex128, order="C")
            self.pbuf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.points_grid.nbytes)

    def manage_buffer(self, height, width):
        obj = self

        class Manager(object):
            def __enter__(self):
                obj.set_arrays(height, width)

            def __exit__(self, *args):
                cl.enqueue_copy(obj.queue, obj.iterations_grid, obj.ibuf).wait()
                cl.enqueue_copy(obj.queue, obj.points_grid, obj.pbuf).wait()
                obj.out = obj.iterations_grid, obj.points_grid

        return Manager()

    def __del__(self):
        if self.ibuf is not None:
            self.ibuf.release()
        if self.pbuf is not None:
            self.pbuf.release()

    def compute(self, *args, **kwargs):
        raise NotImplemented


class ClassicMandelbrotCL(MandelbrotCL):
    def __init__(self):
        super().__init__()
        with open(Path(__file__).parent / 'mandelbrot.cl') as f:
            super().compile(f.read())

    def _compute(self, t_left: mpc, b_right: mpc, width, height, iterations):
        width_per_pix = (b_right.real - t_left.real) / width
        height_per_pix = (-b_right.imag + t_left.imag) / height
        self.prg.pixel64(
            self.queue,
            self.iterations_grid.shape,
            None,
            self.ibuf,
            self.pbuf,
            np.complex128(t_left),
            np.float64(width_per_pix),
            np.float64(height_per_pix),
            np.int32(iterations),
            np.complex128(0),
            np.int32(self.shape[1]),
            np.int32(BREAKOUT_R2)
        )

    def compute(self, config: MandelbrotConfig):
        with self.manage_buffer(config.height, config.width):
            self._compute(config.t_left, config.b_right, config.width, config.height, config.iterations)
        return self.out
