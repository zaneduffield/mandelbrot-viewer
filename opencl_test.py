import pyopencl as cl
import numpy as np
from complex_bf import ComplexBf


def readProgram(filename, ctx):
    lines = open(filename, 'r').read()
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
        self.a, self.abuf = self.set_arrays()

    def set_arrays(self):
        self.a = np.ascontiguousarray(np.zeros((self.width, self.height), dtype=np.int32))
        self.abuf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.a.nbytes)
        # return and assign to avoid pycharm complaints
        return self.a, self.abuf

    def get_pixels(self, t_left: ComplexBf, b_right: ComplexBf, height, width, iterations):
        if height != self.height or width != self.width:
            self.width, self.height = np.int32(width), np.int32(height)
            self.set_arrays()

        center_x = float((b_right.real + t_left.real)/2)
        center_y = float((b_right.imag + t_left.imag)/2)
        width_per_pix = float((b_right.real - t_left.real) / width)
        zoom = 1/width_per_pix
        self.prg.pixel64(self.queue, self.a.shape, None, self.abuf, np.float64(center_x), np.float64(center_y),
                         np.float32(zoom), np.int32(iterations), np.float32(0), np.float32(0), np.int32(self.width), np.int32(self.height))
        cl.enqueue_copy(self.queue, self.a, self.abuf).wait()
        return np.reshape(self.a, (self.width, self.height))

    def __del__(self):
        self.abuf.release()


