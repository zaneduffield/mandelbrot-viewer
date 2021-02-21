import pyopencl as cl
import numpy as np

# import os
# os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
from gmpy2 import mpc


def readProgram(filename, ctx):
    lines = open(filename, "r").read()
    prg = cl.Program(ctx, lines).build()
    return prg


class MandelbrotCL:
    def __init__(self, width, height):
        self.width, self.height = np.int32(width), np.int32(height)
        platform = cl.get_platforms()[0]
        print(platform.get_devices())

        self.mf = cl.mem_flags
        self.ctx = cl.create_some_context(interactive=False)
        self.prg = readProgram("mandelbrot.cl", self.ctx)
        self.queue = cl.CommandQueue(self.ctx)
        self.length = 0

    def set_arrays(self, length):
        self.length = length
        self.a = np.zeros((length, length), dtype=np.uint16, order="C")
        self.abuf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.a.nbytes)

    def get_pixels(self, t_left: mpc, b_right: mpc, height, width, iterations):
        if self.length != max(height, width):
            self.length = max(height, width)
            self.set_arrays(self.length)

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
        cl.enqueue_copy(self.queue, self.a, self.abuf).wait()
        return self.a[:height, :width]

    def __del__(self):
        self.abuf.release()
