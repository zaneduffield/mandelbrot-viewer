import pyopencl as cl
import numpy as np

# import os
# os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
from gmpy2 import mpc


class MandelbrotCL:
    def __init__(self):
        platform = cl.get_platforms()[0]
        print(platform.get_devices())

        self.mf = cl.mem_flags
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.length = 0
        self.prg = None
        self.out = None

    def compile(self, program_contents: str):
        print("compiling...", end='')
        self.prg = cl.Program(self.ctx, program_contents).build()
        print("done")

    def set_arrays(self, height, width):
        if self.length != max(height, width):
            self.length = max(height, width)
            self.a = np.zeros((self.length, self.length), dtype=np.int32, order="C")
            self.abuf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.a.nbytes)

    def manage_buffer(self, height, width):
        cl_class = self

        class Manager(object):
            def __enter__(self):
                cl_class.set_arrays(height, width)

            def __exit__(self, type, value, traceback):
                cl.enqueue_copy(cl_class.queue, cl_class.a, cl_class.abuf).wait()
                cl_class.out = cl_class.a[:height, :width]

        return Manager()

    def _compute_pixels(self, *args, **kwargs):
        raise NotImplemented

    def get_pixels(self, *args, **kwargs):
        raise NotImplemented

    def __del__(self):
        if self.abuf is not None:
            self.abuf.release()


class ClassicMandelbrotCL(MandelbrotCL):
    def __init__(self):
        super().__init__()
        with open('mandelbrot.cl') as f:
            super().compile(f.read())

    def _compute_pixels(self, t_left: mpc, b_right: mpc, width, iterations):
        width_per_pix = float((b_right.real - t_left.real) / width)
        self.prg.pixel64(
            self.queue,
            self.a.shape,
            None,
            self.abuf,
            np.float64(t_left.real),
            np.float64(t_left.imag),
            np.float64(width_per_pix),
            np.int32(iterations),
            np.float32(0),
            np.float32(0),
            np.int32(self.length),
        )

    def get_pixels(self, t_left: mpc, b_right: mpc, height, width, iterations):
        with self.manage_buffer(height, width):
            self._compute_pixels(t_left, b_right, width, iterations)
        return self.out
