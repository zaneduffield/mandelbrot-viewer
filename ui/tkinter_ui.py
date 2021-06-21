import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageTk
from gmpy2 import mpc

from mandelbrot.mandelbrot import Mandelbrot
from mandelbrot_viewer import make_cli_args
from opencl.mandelbrot_cl import PY_OPEN_CL_INSTALLED
from utils.mandelbrot_utils import MandelbrotConfig, set_precision_from_config, my_logger

BROT_COLOUR = (0, 0, 0)
GLITCH_COLOUR = BROT_COLOUR
REF_COLOUR = (255, 0, 0)
MIN_IMAGE_WIDTH = 10
MIN_IMAGE_HEIGHT = 10


class FractalUI(tk.Frame):
    def __init__(
            self,
            parent,
            config: MandelbrotConfig,
            save: bool
    ):
        tk.Frame.__init__(self, parent)
        self.pixels = None
        self.parent = parent
        self.parent.title("Mandelbrot")

        self.perturbation = tk.BooleanVar(value=config.perturbation)
        self.gpu = tk.BooleanVar(value=config.gpu and PY_OPEN_CL_INSTALLED)

        self.start_click = None
        self.rect = None
        self.image_canvas = tk.Canvas(self)
        self.image_canvas.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)
        self.canvas_image = None

        ui = tk.Frame(self)
        ui.pack(side=tk.TOP, fill=tk.X)

        r_controls = tk.Frame(ui)
        l_controls = tk.Frame(ui)
        pertubation_controls = tk.Frame(ui)

        r_controls.pack(side=tk.RIGHT)
        pertubation_controls.pack(side=tk.RIGHT)
        l_controls.pack(side=tk.LEFT)

        tk.Button(l_controls, command=self.recompute, text="recompute").pack(side=tk.RIGHT)

        tk.Label(l_controls, text="Max iterations", height=1).pack(side=tk.LEFT)
        self.iterations = tk.StringVar(value=config.iterations)
        iter_entry = tk.Entry(l_controls, textvariable=self.iterations, width=10)
        iter_entry.bind("<Return>", self.on_iter_submit)
        iter_entry.pack(side=tk.LEFT)

        tk.Button(r_controls, command=self.copy_cli, text="copy CLI").pack(side=tk.RIGHT)
        self.mag = tk.Text(r_controls, height=1, width=20, state=tk.DISABLED)
        self.mag.pack(side=tk.RIGHT)
        tk.Button(r_controls, command=self.reset, text="reset").pack(side=tk.RIGHT)
        tk.Button(r_controls, command=self.back, text="prev").pack(side=tk.RIGHT)
        tk.Button(r_controls, command=self.next, text="next").pack(side=tk.RIGHT)
        tk.Button(r_controls, command=self.recolour, text="recolour").pack(side=tk.RIGHT)
        tk.Checkbutton(
            pertubation_controls, text="high precision", variable=self.perturbation
        ).pack(side=tk.RIGHT)
        gpu_checkbox_state = tk.NORMAL if PY_OPEN_CL_INSTALLED else tk.DISABLED
        tk.Checkbutton(r_controls, text="gpu", variable=self.gpu, state=gpu_checkbox_state).pack(side=tk.RIGHT)

        self.pack(fill=tk.BOTH, expand=True)

        self.image_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.image_canvas.bind("<B1-Motion>", self.on_move_press)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.fractal = Mandelbrot()
        self.compute_config = config
        self.save = save
        config.t_left, config.b_right = self.expand_to_match_screen_ratio(config.t_left, config.b_right)
        self.set_corners(config.t_left, config.b_right)
        self.palette = generate_palette()
        self.compute_and_draw()

    def copy_cli(self):
        self.parent.clipboard_clear()
        self.parent.clipboard_append(make_cli_args(self.compute_config))

    def recolour(self):
        self.palette = generate_palette()
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
        self.mag.config(state=tk.NORMAL)
        width = self.compute_config.b_right.real - self.compute_config.t_left.real
        self.mag.replace("1.0", "1.end", f"width: {width:.2e}")
        self.mag.config(state=tk.DISABLED)

    def on_iter_submit(self, event):
        self.palette = generate_palette()
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
        self.image_canvas.delete(self.rect)
        self.rect = None
        self.compute_and_draw()

    def reposition(self, t_left_coords: tuple, b_right_coords: tuple):
        b = self.compute_config.b_right
        t = self.compute_config.t_left

        hor_scale = (b.real - t.real) / self.compute_config.width
        ver_scale = (t.imag - b.imag) / self.compute_config.height

        t_left = t + hor_scale * t_left_coords[0] - ver_scale * t_left_coords[1] * 1j
        b_right = t + hor_scale * b_right_coords[0] - ver_scale * b_right_coords[1] * 1j
        t_left, b_right = self.expand_to_match_screen_ratio(t_left, b_right)
        self.set_corners(t_left, b_right)

        t_left_diag, b_right_diag = t_left - t, b_right - t
        # self.crop_and_upscale(t_left_diag, b_right_diag, hor_scale, ver_scale)

    def expand_to_match_screen_ratio(self, t_left: mpc, b_right: mpc):
        height = t_left.imag - b_right.imag
        width = b_right.real - t_left.real

        ratio_target = self.compute_config.height / self.compute_config.width
        inverse_ratio_target = self.compute_config.width / self.compute_config.height
        ratio_curr = height / width

        if ratio_target > ratio_curr:
            diff = (width * ratio_target - height) / 2
            t_left += diff * 1j
            b_right -= diff * 1j
        else:
            diff = (height * inverse_ratio_target - width) / 2
            t_left -= diff
            b_right += diff

        return t_left, b_right

    def set_corners(self, t_left: mpc, b_right: mpc):
        self.compute_config.t_left, self.compute_config.b_right = t_left, b_right
        self.set_mag()
        set_precision_from_config(self.compute_config)

    def recompute(self):
        self.fractal.back()
        self.compute_and_draw()

    def compute_and_draw(self):
        threading.Thread(target=self.threaded_compute_and_draw).start()

    def threaded_compute_and_draw(self):
        my_logger.info("-" * 80)
        start = time.time()
        self.write_config()
        for pixels in self.fractal.get_pixels(self.compute_config):
            self.pixels = pixels
            self.draw_pixels(self.pixels)

        duration = time.time() - start
        my_logger.info("computation took {} seconds".format(round(duration, 2)))
        my_logger.info("-" * 80)

    def crop_and_upscale(self, t_left_diag: mpc, b_right_diag: mpc, hor_scale, ver_scale):
        t_left_expanded_coords = (round(t_left_diag.real / hor_scale), round(-t_left_diag.imag / ver_scale))
        b_right_expanded_coords = (round(b_right_diag.real / hor_scale), round(-b_right_diag.imag / ver_scale))
        w, h = self.image.size
        image = self.image.crop((*t_left_expanded_coords, *b_right_expanded_coords))
        image = image.resize((w, h), resample=Image.BOX)
        self.set_image(image)

    def set_image(self, image):
        self.image = image
        if self.save:
            self.save_image(image)

        tk_image = ImageTk.PhotoImage(image)
        if self.canvas_image is None:
            self.canvas_image = self.image_canvas.create_image(
                0, 0, image=tk_image, anchor=tk.NW
            )
        else:
            self.image_canvas.itemconfig(self.canvas_image, image=tk_image)
        self.image_canvas.image = tk_image

    def draw_pixels(self, iterations, ncycles=30):
        MAX_VAL = 255
        scaled_iterations = np.sqrt(np.abs(iterations)) % ncycles / ncycles
        num_colours = self.palette.shape[0] - 1
        indexes = np.round(scaled_iterations * num_colours).astype(np.int32)
        colours = (MAX_VAL * self.palette[indexes]).astype(np.uint8)
        self.set_image(Image.fromarray(colours, "RGB"))

    def save_image(self, image):
        image.save(
            "{}.png".format(time.strftime("%Y.%m.%d-%H.%M.%S")),
            "PNG",
            optimize=True,
        )


def generate_palette(rgb_offsets=(0.1, -0.2, 0.5)):
    cols = np.linspace(0, 1, 2**11)
    a = np.column_stack((cols + offset) * 2 * np.pi for offset in rgb_offsets)
    return 0.5 + 0.5 * np.cos(a)


def run(config: MandelbrotConfig, save: bool):
    root = tk.Tk()
    root.iconbitmap(Path(__file__).parent / "brot.ico")
    screen_size_prop = 0.6
    config.height = round(root.winfo_screenheight() * screen_size_prop)
    config.width = round(root.winfo_screenwidth() * screen_size_prop)

    set_precision_from_config(config)
    FractalUI(
        parent=root,
        config=config,
        save=save
    )
    root.geometry("{}x{}".format(config.width, config.height))
    root.mainloop()
