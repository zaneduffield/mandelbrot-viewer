import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageTk
from gmpy2 import mpc, mpfr

from mandelbrot.mandelbrot import Mandelbrot
from mandelbrot_viewer import make_cli_args
from opencl.mandelbrot_cl import PY_OPEN_CL_INSTALLED
from utils.constants import BREAKOUT_R2, GLITCH_ITER
from utils.mandelbrot_utils import MandelbrotConfig, my_logger

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
        self.compute_result = None
        self.parent = parent
        self.parent.title("Mandelbrot")

        top_ui = tk.Frame(self)
        bottom_ui = tk.Frame(self)
        top_ui.pack(side=tk.TOP, fill=tk.X)
        bottom_ui.pack(side=tk.BOTTOM, fill=tk.X)

        tr_controls = tk.Frame(top_ui)
        tl_controls = tk.Frame(top_ui)
        pertubation_controls = tk.Frame(top_ui)

        self.perturbation = tk.BooleanVar(value=config.perturbation)
        self.gpu = tk.BooleanVar(value=config.gpu and PY_OPEN_CL_INSTALLED)

        self.start_click = None
        self.rect = None
        self.image_canvas = tk.Canvas(self)
        self.image_canvas.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)
        self.canvas_image = None

        tr_controls.pack(side=tk.RIGHT)
        tl_controls.pack(side=tk.LEFT)
        pertubation_controls.pack(side=tk.LEFT)

        tk.Button(tl_controls, command=self.recompute, text="recompute").pack(side=tk.RIGHT)

        self.zoom = tk.StringVar()
        self.write_zoom_entry(config.get_zoom())
        self.zoom_entry = tk.Entry(tl_controls, textvariable=self.zoom, width=10, bd=3)
        self.zoom_entry.bind("<Return>", lambda _: self.compute_and_draw())
        self.zoom_entry.pack(side=tk.LEFT)

        tk.Label(tl_controls, text="Max iterations", height=1).pack(side=tk.LEFT)
        self.max_iterations = tk.StringVar(value=config.max_iterations)
        iter_entry = tk.Entry(tl_controls, textvariable=self.max_iterations, width=13, bd=3)
        iter_entry.bind("<Return>", lambda _: self.compute_and_draw())
        iter_entry.pack(side=tk.LEFT)

        tk.Button(tr_controls, command=self.copy_cli, text="copy CLI").pack(side=tk.RIGHT)


        tk.Button(bottom_ui, command=self.reset, text="reset").pack(side=tk.RIGHT)
        tk.Button(bottom_ui, command=self.back, text="prev").pack(side=tk.RIGHT)
        tk.Button(bottom_ui, command=self.next, text="next").pack(side=tk.RIGHT)
        tk.Button(bottom_ui, command=self.recolour, text="recolour").pack(side=tk.LEFT)
        tk.Checkbutton(
            pertubation_controls, text="high precision", variable=self.perturbation
        ).pack(side=tk.RIGHT)
        gpu_checkbox_state = tk.NORMAL if PY_OPEN_CL_INSTALLED else tk.DISABLED
        tk.Checkbutton(tl_controls, text="gpu", variable=self.gpu, state=gpu_checkbox_state).pack(side=tk.RIGHT)

        self.pack(fill=tk.BOTH, expand=True)

        self.image_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.image_canvas.bind("<B1-Motion>", self.on_move_press)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.fractal = Mandelbrot()
        self.compute_config = config
        self.save = save
        self.palette = generate_palette()
        self.computing = False
        self.compute_and_draw()

    def copy_cli(self):
        self.parent.clipboard_clear()
        self.parent.clipboard_append(make_cli_args(self.compute_config))

    def recolour(self):
        self.palette = generate_palette()
        self.draw(self.compute_result)

    def read_config(self):
        self.max_iterations.set(self.compute_config.max_iterations)
        self.perturbation.set(self.compute_config.perturbation)
        self.gpu.set(self.compute_config.gpu)
        self.write_zoom_entry(self.compute_config.get_zoom())

    def write_config(self):
        self.compute_config.max_iterations = int(self.max_iterations.get())
        self.compute_config.perturbation = self.perturbation.get()
        self.compute_config.gpu = self.gpu.get()
        self.set_zoom()
        width = max(MIN_IMAGE_WIDTH, self.image_canvas.winfo_width())
        height = max(MIN_IMAGE_HEIGHT, self.image_canvas.winfo_height())
        self.compute_config.set_image_dimensions(width, height)

    def set_zoom(self):
        # validate zoom
        zoom_text = self.zoom.get()
        self.compute_config.set_zoom(mpfr(zoom_text))

    def reset(self):
        pop = None
        temp = self.fractal.back()
        while temp is not None:
            pop = temp
            temp = self.fractal.back()

        self.load(pop)

    def load(self, pop: Tuple[MandelbrotConfig, Tuple]):
        if pop is None:
            return
        self.compute_config, self.compute_result = pop
        self.read_config()
        self.draw(self.compute_result)

    def back(self):
        self.load(self.fractal.back())

    def next(self):
        self.load(self.fractal.next())

    def write_zoom_entry(self, zoom: mpfr):
        self.zoom.set(f"{zoom:.3e}")

    def on_button_press(self, event):
        self.start_click = event

    def on_move_press(self, event):
        # expand rectangle as you drag the mouse
        dw = abs(event.x - self.start_click.x)
        dh = round(dw * self.compute_config.image_height / self.compute_config.image_width)
        t_left = (self.start_click.x - dw, self.start_click.y - dh)
        b_right = (self.start_click.x + dw, self.start_click.y + dh)
        rect_coords = (*t_left, *b_right)
        if self.rect is None:
            self.rect = self.image_canvas.create_rectangle(
                *rect_coords, fill=""
            )
        else:
            self.image_canvas.coords(self.rect, *rect_coords)

    def on_button_release(self, event):
        if self.rect is None:
            return

        coords = self.image_canvas.coords(self.rect)
        new_pix_width = coords[2] - coords[0]

        canvas_center_x = self.image_canvas.winfo_width() / 2
        canvas_center_y = self.image_canvas.winfo_height() / 2
        diff = (self.start_click.x - canvas_center_x) + 1j * (canvas_center_y - self.start_click.y)
        width_per_pix = self.compute_config.get_width() / self.image_canvas.winfo_width()
        self.compute_config.set_center(self.compute_config.get_center() + width_per_pix * diff)
        self.compute_config.set_zoom_from_width(new_pix_width * width_per_pix)

        self.image_canvas.delete(self.rect)
        self.write_zoom_entry(self.compute_config.get_zoom())
        self.rect = None
        self.compute_and_draw()

    def recompute(self):
        self.fractal.back()
        self.compute_and_draw()

    def compute_and_draw(self):
        if self.computing:
            return
        threading.Thread(target=self.threaded_compute_and_draw).start()

    def threaded_compute_and_draw(self):
        self.computing = True
        my_logger.info("-" * 80)
        start = time.time()
        self.write_config()
        for result in self.fractal.compute(self.compute_config):
            self.compute_result = result
            self.draw(self.compute_result)

        duration = time.time() - start
        my_logger.info("computation took {} seconds".format(round(duration, 2)))
        my_logger.info("-" * 80)
        self.computing = False

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

    def draw(self, result):
        iterations, points = result
        # iterations = iterations[:]  # to prevent mutation

        # make perturbation glitches appear like brot points
        iterations = iterations + (iterations == GLITCH_ITER).astype(np.int32) * self.compute_config.max_iterations
        brot_pixels = iterations == self.compute_config.max_iterations

        iterations = convert_to_fractional_counts(iterations, points).astype(np.int64)
        # rescale to 0 so we can use the values as indices to the cumulative counts
        iterations -= np.min(iterations)

        histogram = np.histogram(iterations, np.max(iterations) + 1)[0]
        # don't let brot pixels affect colour scaling
        histogram[-1] -= np.sum(brot_pixels)
        cumulative_counts = np.cumsum(histogram)
        # rescale so the entire colour range is used (otherwise the first colour used would be offset)
        cumulative_counts = cumulative_counts - cumulative_counts[0]
        relative_cumulative_counts = cumulative_counts[iterations] / max(1, cumulative_counts[-1])

        num_colours = self.palette.shape[0] - 1
        indices = np.minimum(num_colours, (num_colours * relative_cumulative_counts).astype(np.int32))

        colours = (255 * self.palette[indices]).astype(np.uint8)
        colours[brot_pixels] = BROT_COLOUR

        self.set_image(Image.fromarray(colours, "RGB"))

    def save_image(self, image):
        image.save(
            "{}.png".format(time.strftime("%Y.%m.%d-%H.%M.%S")),
            "PNG",
            optimize=True,
        )


def convert_to_fractional_counts(iterations, points, scale=100):
    abs_points = np.maximum(2, np.abs(points))
    return scale * (iterations + np.log2(0.5*np.log2(BREAKOUT_R2)) - np.log2(np.log2(abs_points)))


def generate_palette(rgb_offsets=None):
    if rgb_offsets is None:
        rgb_offsets = np.random.random(size=3)
    cols = np.linspace(0, 1, 2**11)
    a = np.column_stack([(cols + offset) * 1.5 * np.pi for offset in rgb_offsets])
    return 0.5 + 0.5 * np.cos(a)


def run(config: MandelbrotConfig, save: bool):
    root = tk.Tk()
    root.iconbitmap(Path(__file__).parent / "brot.ico")

    if 0 in [config.image_height, config.image_width]:
        screen_size_prop = 0.4
        config.set_image_dimensions(
            round(root.winfo_screenwidth() * screen_size_prop), round(root.winfo_screenheight() * screen_size_prop)
        )

    FractalUI(
        parent=root,
        config=config,
        save=save
    )
    root.geometry("{}x{}".format(config.image_width, config.image_height))
    root.mainloop()
