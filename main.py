import cProfile
import os
import argparse
import pstats
import sys
import math
from typing import Tuple

import numpy as np
import time
import random
import threading
from tkinter import *
from PIL import Image, ImageTk

from mandelbrot import Mandelbrot
from mandelbrot_utils import MandelbrotConfig
from gmpy2 import mpc, get_context

BROT_COLOUR = (0, 0, 0)
GLITCH_COLOUR = BROT_COLOUR
REF_COLOUR = (255, 0, 0)
MIN_IMAGE_WIDTH = 10
MIN_IMAGE_HEIGHT = 10


class FractalUI(Frame):
    def __init__(
        self,
        parent,
        height,
        width,
        t_left: mpc,
        b_right: mpc,
        iterations=None,
        save=False,
        use_multiprocessing: bool = True,
        use_gpu: bool = False,
        perturbation: bool = False,
        num_series_terms: int = 10,
    ):
        Frame.__init__(self, parent)
        self.pixels = None
        self.parent = parent
        self.parent.title("Mandelbrot")

        self.multiprocessing = BooleanVar(value=use_multiprocessing)
        self.perturbation = BooleanVar(value=perturbation)
        self.gpu = BooleanVar(value=use_gpu)

        self.start_click = None
        self.rect = None
        self.image_canvas = Canvas(self)
        self.image_canvas.pack(side=BOTTOM, expand=True, fill=BOTH)
        self.canvas_image = None

        ui = Frame(self)
        ui.pack(side=TOP, fill=X)
        r_controls = Frame(ui)
        r_controls.pack(side=RIGHT)
        pertubation_controls = Frame(ui)
        pertubation_controls.pack(side=RIGHT)
        l_controls = Frame(ui)
        l_controls.pack(side=LEFT)

        Button(l_controls, command=self.copy_cli, text="copy coords").pack(
            side=RIGHT
        )
        Button(l_controls, command=self.recompute, text="recompute").pack(side=RIGHT)

        Label(l_controls, text="Max iterations", height=1).pack(side=LEFT)
        self.iterations = StringVar(value=iterations)
        iter_entry = Entry(l_controls, textvariable=self.iterations, width=10)
        iter_entry.bind("<Return>", self.on_iter_submit)
        iter_entry.pack(side=LEFT)

        self.mag = Text(r_controls, height=1, width=20, state=DISABLED)
        self.mag.pack(side=RIGHT)
        Button(r_controls, command=self.reset, text="reset").pack(side=RIGHT)
        Button(r_controls, command=self.back, text="back").pack(side=RIGHT)
        Button(r_controls, command=self.next, text="next").pack(side=RIGHT)
        Button(r_controls, command=self.recolour, text="recolour").pack(side=RIGHT)
        Checkbutton(
            pertubation_controls, text="high precision", variable=self.perturbation
        ).pack(side=RIGHT)
        Checkbutton(
            r_controls, text="multiprocessing", variable=self.multiprocessing
        ).pack(side=RIGHT)
        Checkbutton(r_controls, text="gpu", variable=self.gpu).pack(side=RIGHT)

        self.pack(fill=BOTH, expand=True)

        self.image_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.image_canvas.bind("<B1-Motion>", self.on_move_press)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.fractal = Mandelbrot()
        self.compute_config = MandelbrotConfig(
            width=width,
            height=height,
            t_left=t_left,
            b_right=b_right,
            iterations=iterations,
            perturbation=perturbation,
            gpu=use_gpu,
            num_series_terms=num_series_terms,
        )
        self.set_corners(t_left, b_right)
        self.palette = generate_palette(self.compute_config.iterations)
        self.save = save
        self.compute_and_draw()

    def copy_cli(self):
        self.parent.clipboard_clear()
        self.parent.clipboard_append(
            f'-tlr " {str(self.compute_config.t_left.real)}" '
            f'-tli " {str(self.compute_config.t_left.imag)}" '
            f'-bri " {str(self.compute_config.b_right.imag)}" '
            f'-brr " {str(self.compute_config.b_right.real)}" '
            f"-i {self.compute_config.iterations}"
        )
        if self.compute_config.perturbation:
            self.parent.clipboard_append(" -p")

        if self.compute_config.gpu:
            self.parent.clipboard_append(" -g")

    def recolour(self):
        self.palette = generate_palette(self.compute_config.iterations)
        self.draw_pixels(self.pixels)

    def read_config(self):
        self.iterations.set(self.compute_config.iterations)
        self.perturbation.set(self.compute_config.perturbation)
        self.gpu.set(self.compute_config.gpu)

    def write_config(self):
        self.compute_config.iterations = int(self.iterations.get())
        self.compute_config.perturbation = self.perturbation.get()
        self.compute_config.gpu = self.gpu.get()
        width = self.image_canvas.winfo_width()
        if width > MIN_IMAGE_WIDTH:
            self.compute_config.width = width
        height = self.image_canvas.winfo_height()
        if height > MIN_IMAGE_HEIGHT:
            self.compute_config.height = height

    def reset(self):
        pop = None
        temp = self.fractal.back()
        while temp is not None:
            pop = temp
            temp = self.fractal.back()

        self.load(pop)

    def load(self, pop: Tuple[MandelbrotConfig, np.array]):
        if pop is None:
            return
        self.compute_config, self.pixels = pop
        self.read_config()
        self.draw_pixels(self.pixels)

    def back(self):
        self.load(self.fractal.back())

    def next(self):
        self.load(self.fractal.next())

    def set_mag(self):
        self.mag.config(state=NORMAL)
        width = self.compute_config.b_right.real - self.compute_config.t_left.real
        self.mag.replace("1.0", "1.end", f"width: {width:.2e}")
        self.mag.config(state=DISABLED)

    def on_iter_submit(self, event):
        self.compute_config.iterations = int(self.iterations.get())
        self.palette = generate_palette(self.compute_config.iterations)
        self.compute_and_draw()

    def on_button_press(self, event):
        self.start_click = (event.x, event.y)

    def on_move_press(self, event):
        # expand rectangle as you drag the mouse
        if self.rect is None:
            self.rect = self.image_canvas.create_rectangle(
                self.start_click[0], self.start_click[1], 1, 1, fill=""
            )

        self.image_canvas.coords(
            self.rect, self.start_click[0], self.start_click[1], event.x, event.y
        )

    def on_button_release(self, event):
        if self.rect is None:
            return

        x = self.image_canvas.coords(self.rect)
        t_left_coords = (min(x[0], x[2]), min(x[1], x[3]))
        b_right_coords = (max(x[0], x[2]), max(x[1], x[3]))

        self.reposition(t_left_coords, b_right_coords)
        self.set_mag()
        self.image_canvas.delete(self.rect)
        self.rect = None
        self.compute_and_draw()

    def reposition(self, t_left_coords: tuple, b_right_coords: tuple):
        b = self.compute_config.b_right
        t = self.compute_config.t_left

        hor_scale = (b.real - t.real) / self.compute_config.width
        ver_scale = (t.imag - b.imag) / self.compute_config.height

        t_left = t + mpc(
            hor_scale * t_left_coords[0] - ver_scale * t_left_coords[1] * 1j
        )
        b_right = t + mpc(
            hor_scale * b_right_coords[0] - ver_scale * b_right_coords[1] * 1j
        )
        self.set_corners(t_left, b_right)

    def set_corners(self, t_left: mpc, b_right: mpc):
        height = t_left.imag - b_right.imag
        width = b_right.real - t_left.real

        get_context().precision = int(
            -np.log2(float(width / self.compute_config.width))
        )
        ratio_target = self.compute_config.height / self.compute_config.width
        ratio_curr = height / width

        if ratio_target > ratio_curr:
            diff = (width * ratio_target - height) / 2
            t_left += diff * 1j
            b_right -= diff * 1j
        else:
            diff = (height / ratio_target - width) / 2
            t_left -= diff
            b_right += diff

        self.compute_config.t_left, self.compute_config.b_right = t_left, b_right

    def recompute(self):
        self.fractal.back()
        self.compute_and_draw()

    def compute_and_draw(self):
        threading.Thread(target=self.threaded_compute_and_draw).start()

    def threaded_compute_and_draw(self):
        print("-" * 80)
        start = time.time()
        self.write_config()
        for pixels in self.fractal.get_pixels(self.compute_config):
            self.pixels = pixels
            self.draw_pixels(self.pixels)

        comp_time = time.time()
        print("computation took {} seconds".format(round(comp_time - start, 2)))

    def set_image(self, image):
        if self.canvas_image is None:
            self.canvas_image = self.image_canvas.create_image(
                0, 0, image=image, anchor=NW
            )
        else:
            self.image_canvas.itemconfig(self.canvas_image, image=image)

    def draw_pixels(self, pixels):
        img = Image.fromarray(self.palette[pixels], "RGB")
        if self.save:
            self.save_image(img)
        self.image = ImageTk.PhotoImage(img)
        self.set_image(self.image)

    def save_image(self, img):
        img.save(
            "output/{}.png".format(time.strftime("%Y-%m-%d-%H:%M:%S")),
            "PNG",
            optimize=True,
        )


def generate_palette(iterations):
    palette_length = iterations // 2

    pairs = [
        (
            2 * math.pi / (rand_b * (palette_length // 2) + palette_length // 2),
            256 * rand_c,
        )
        for rand_b, rand_c in np.random.random((3, 2))
    ]

    palette = np.empty((iterations + 3, 3), dtype=np.uint8)
    palette[0] = BROT_COLOUR
    palette[iterations + 1] = REF_COLOUR
    palette[iterations + 2] = GLITCH_COLOUR

    for i, (b, c) in enumerate(pairs):
        iter_range = np.arange(1, iterations + 1) % palette_length
        temp = 128 * (np.sin(b * iter_range + c) + 1)
        palette[1 : iterations + 1, i] = np.maximum(
            0, np.minimum(temp.astype(int), 255)
        )

    return palette


def main():
    master = Tk()
    height = round(master.winfo_screenheight() * 0.6)
    width = height*2
    parser = argparse.ArgumentParser(description="Generate the Mandelbrot set")
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="The number of iterations done for each pixel.",
        default=500,
    )
    parser.add_argument(
        "-tlr",
        "--top-left-real",
        type=str,
        help="The top-left real coordinate of the area to render in str form",
        default="-1.5",
    )
    parser.add_argument(
        "-tli",
        "--top-left-imag",
        type=str,
        help="The top-left imag coordinate of the area to render in str form",
        default="1.25",
    )
    parser.add_argument(
        "-brr",
        "--bottom-right-real",
        type=str,
        help="The bottom-right real coordinate of the area to render in str form",
        default="0.5",
    )
    parser.add_argument(
        "-bri",
        "--bottom-right-imag",
        type=str,
        help="The bottom-right imag coordinate of the area to render in str form",
        default="-1.25",
    )
    parser.add_argument("-w", "--width", type=int, help="The width of the image.")
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save the generated image."
    )
    parser.add_argument(
        "-nm",
        "--no-multiprocessing",
        action="store_false",
        help="Don't use multiprocessing.",
    )
    parser.add_argument(
        "-p",
        "--perturbation",
        action="store_true",
        help="Use perturbation theory for high precision " "computation.",
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Use GPU via opencl to render"
    )
    args = parser.parse_args()

    get_context().precision = 3000

    t_left = mpc(f"({args.top_left_real} {args.top_left_imag})")
    b_right = mpc(f"({args.bottom_right_real} {args.bottom_right_imag})")
    FractalUI(
        parent=master,
        height=height,
        width=width,
        use_multiprocessing=args.no_multiprocessing,
        t_left=t_left,
        b_right=b_right,
        iterations=args.iterations,
        save=args.save,
        perturbation=args.perturbation,
        use_gpu=args.gpu,
    )

    master.geometry("{}x{}".format(width, height))
    master.mainloop()


if __name__ == "__main__":

    # profile = cProfile.Profile()
    # profile.runcall(main)
    # ps = pstats.Stats(profile)
    # ps.sort_stats("cumtime")
    # ps.print_stats(30)
    main()
