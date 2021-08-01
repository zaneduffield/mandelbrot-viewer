from functools import lru_cache
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

        self._compile_cache = {}
        for double_precision in [True, False]:
            self._double_precision = double_precision
            self.compile()

    def get_program_contents(self):
        raise NotImplemented

    def set_precision(self, double_precision):
        if self._double_precision == double_precision:
            return
        self._double_precision = double_precision
        self.shape = None
        self.compile()

    def _get_complex_dtype(self):
        return np.complex128 if self._double_precision else np.complex64

    def _get_real_dtype(self):
        return np.float64 if self._double_precision else np.float32

    def _get_cl_program(self, double_precision):
        if double_precision not in self._compile_cache:
            program_contents = self.get_program_contents()
            if double_precision:
                program_contents = program_contents.replace("float", "double")

            prg = cl.Program(self.ctx, program_contents).build()
            self._compile_cache[double_precision] = prg
        return self._compile_cache[double_precision]

    def compile(self):
        my_logger.debug("compiling...")
        self.prg = self._get_cl_program(self._double_precision)
        my_logger.debug("done")

    def set_arrays(self, height, width):
        if self.shape != (height, width):
            self.shape = (height, width)
            self.iterations_grid = np.zeros(self.shape, dtype=np.int32, order="C")
            self.ibuf = cl.Buffer(
                self.ctx, self.mf.WRITE_ONLY, self.iterations_grid.nbytes
            )
            self.points_grid = np.zeros(
                self.shape, dtype=self._get_complex_dtype(), order="C"
            )
            self.pbuf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.points_grid.nbytes)

    def manage_buffer(self, height, width):
        obj = self

        class Manager(object):
            def __enter__(self):
                obj.set_arrays(height, width)

            def __exit__(self, *args):
                cl.enqueue_copy(obj.queue, obj.iterations_grid, obj.ibuf)
                cl.enqueue_copy(obj.queue, obj.points_grid, obj.pbuf)
                obj.queue.finish()
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
    def get_program_contents(self):
        with open(Path(__file__).parent / "mandelbrot.cl") as f:
            return f.read()

    def _compute(self, t_left: mpc, b_right: mpc, width, height, iterations):
        width_per_pix = (b_right.real - t_left.real) / width
        height_per_pix = (-b_right.imag + t_left.imag) / height
        complex_dtype = self._get_complex_dtype()
        real_dtype = self._get_real_dtype()
        self.prg.pixel64(
            self.queue,
            self.iterations_grid.shape,
            None,
            self.ibuf,
            self.pbuf,
            complex_dtype(t_left),
            real_dtype(width_per_pix),
            real_dtype(height_per_pix),
            np.int32(iterations),
            complex_dtype(0),
            np.int32(self.shape[1]),
            np.int32(BREAKOUT_R2),
        )

    def compute(self, config: MandelbrotConfig):
        self.set_precision(config.gpu_double_precision)
        with self.manage_buffer(config.image_height, config.image_width):
            self._compute(
                config.t_left(),
                config.b_right(),
                config.image_width,
                config.image_height,
                config.max_iterations,
            )
        return self.out
