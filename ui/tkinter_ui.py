import threading
import time
import tkinter as tk
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageTk
from gmpy2 import mpc, mpfr
from tkinter.filedialog import asksaveasfile

from mandelbrot.mandelbrot import Mandelbrot, convert_to_fractional_counts
from mandelbrot_viewer import make_cli_args
from opencl.mandelbrot_cl import PY_OPEN_CL_INSTALLED
from ui.colouring import histogram_colouring, generate_palette
from utils.constants import BREAKOUT_R2, GLITCH_ITER
from utils.mandelbrot_utils import MandelbrotConfig, my_logger

MIN_WINDOW_WIDTH = 400
MIN_WIDOW_HEIGHT = 300


@contextmanager
def temp_disable(parent):
    widgets_set = set_state_recursive(enumerate_leaves(parent), tk.NORMAL, tk.DISABLED)
    try:
        yield
    finally:
        set_state_recursive(widgets_set, tk.DISABLED, tk.NORMAL)


def enumerate_leaves(parent):
    if parent.winfo_class() in ('Frame', 'Labelframe'):
        out = []
        for child in parent.winfo_children():
            out += enumerate_leaves(child)
        return out
    return [parent]


def set_state_recursive(widgets, state_from, state_to):
    out = []
    for widget in widgets:
        states = widget.config().get('state')
        if states is None or state_from in states:
            widget.configure(state=state_to)
            out.append(widget)
    return out


class FractalUI(tk.Frame):
    def __init__(
            self,
            parent,
            config: MandelbrotConfig,
    ):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Mandelbrot")

        self.fractal = Mandelbrot()
        self.compute_config = config
        self.palette = generate_palette()
        self.computing = False
        self.compute_result = None

        ###########################################################################################

        self.control_panel = tk.Frame(self)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)

        compute_controls = tk.Frame(self.control_panel)
        gpu_controls = tk.LabelFrame(self.control_panel)
        perturbation_controls = tk.LabelFrame(self.control_panel)
        image_controls = tk.LabelFrame(self.control_panel)
        resolution_controls = tk.LabelFrame(self.control_panel, text="resolution scale")
        navigation_controls = tk.LabelFrame(self.control_panel)

        panel_padding = 7
        compute_controls.pack(side=tk.TOP, fill=tk.X, pady=panel_padding)
        gpu_controls.pack(side=tk.TOP, fill=tk.X, pady=panel_padding)
        perturbation_controls.pack(side=tk.TOP, fill=tk.X, pady=panel_padding)
        navigation_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=panel_padding)
        resolution_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=panel_padding)
        image_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=panel_padding)

        ###########################################################################################

        center_frame = tk.LabelFrame(compute_controls, text="Center")
        self.center_re = tk.StringVar()
        self.center_im = tk.StringVar()
        self.write_center_entry(config.get_center())
        center_re = tk.Frame(center_frame)
        center_im = tk.Frame(center_frame)
        tk.Label(center_re, text="Re:").pack(side=tk.LEFT)
        tk.Label(center_im, text="Im:").pack(side=tk.LEFT)
        tk.Entry(center_re, textvariable=self.center_re, state='readonly').pack(side=tk.LEFT, fill=tk.X)
        tk.Entry(center_im, textvariable=self.center_im, state='readonly').pack(side=tk.LEFT, fill=tk.X)
        center_re.pack(side=tk.TOP, fill=tk.X)
        center_im.pack(side=tk.TOP, fill=tk.X)

        self.zoom_frame = tk.LabelFrame(compute_controls, text="Zoom")
        self.zoom = tk.StringVar()
        zoom_entry = tk.Entry(self.zoom_frame, textvariable=self.zoom)
        self.write_zoom_entry(config.get_zoom())
        tk.Button(self.zoom_frame, text="+", command=self.increase_zoom).pack(side=tk.RIGHT)
        tk.Button(self.zoom_frame, text="-", command=self.decrease_zoom).pack(side=tk.RIGHT)

        iterations_frame = tk.LabelFrame(compute_controls, text="Max Iterations")
        self.max_iterations = tk.StringVar(value=config.max_iterations)
        self.iter_entry = tk.Entry(iterations_frame, textvariable=self.max_iterations, width=13)

        for entry in [zoom_entry, self.iter_entry]:
            entry.pack(fill=tk.X)
            entry.bind("<Return>", lambda _: self.compute_and_draw())

        other = tk.Frame(compute_controls)
        tk.Button(other, command=self.recompute, text="recompute").pack(side=tk.LEFT)
        tk.Button(other, command=self.copy_cli, text="copy CLI").pack(side=tk.RIGHT)
        other.pack(side=tk.TOP, pady=5)

        center_frame.pack(side=tk.TOP, fill=tk.X)
        self.zoom_frame.pack(side=tk.TOP, fill=tk.X)
        iterations_frame.pack(side=tk.TOP, fill=tk.X)

        ###########################################################################################

        self.gpu = tk.BooleanVar(value=config.gpu and PY_OPEN_CL_INSTALLED)
        self.gpu_double_precision = tk.BooleanVar(value=config.gpu_double_precision)
        gpu_checkbox_state = tk.NORMAL if PY_OPEN_CL_INSTALLED else tk.DISABLED

        gpu_double_precision_checkbox = tk.Checkbutton(
            gpu_controls,
            text="gpu double precision",
            variable=self.gpu_double_precision,
        )
        _set_double_precision_state = (
            lambda: gpu_double_precision_checkbox.configure(
                state=tk.NORMAL if self.gpu.get() else tk.DISABLED
            )
        )
        _set_double_precision_state()

        tk.Checkbutton(
            gpu_controls,
            text="gpu",
            variable=self.gpu,
            state=gpu_checkbox_state,
            command=_set_double_precision_state,
        ).grid(sticky=tk.W)
        gpu_double_precision_checkbox.grid(row=1, sticky=tk.W)

        ###########################################################################################

        self.perturbation = tk.BooleanVar(value=config.perturbation)
        tk.Checkbutton(
            perturbation_controls, text="high precision", variable=self.perturbation
        ).pack(side=tk.LEFT)

        ###########################################################################################

        self.resolution_scale = tk.DoubleVar(value=1)
        tk.Scale(resolution_controls, variable=self.resolution_scale, from_=0.2, to=1.5, orient=tk.HORIZONTAL,
                 resolution=0.2, command=lambda _: self.compute_and_draw()).pack(fill=tk.X)

        ###########################################################################################

        tk.Button(image_controls, command=self.recolour, text="recolour").pack(side=tk.LEFT)
        tk.Button(image_controls, command=self.save_image, text="save").pack(side=tk.RIGHT)

        ###########################################################################################

        tk.Button(navigation_controls, command=self.back, text="prev").pack(side=tk.LEFT)
        tk.Button(navigation_controls, command=self.next, text="next").pack(side=tk.LEFT)
        tk.Button(navigation_controls, command=self.reset, text="reset").pack(side=tk.RIGHT)

        ###########################################################################################

        self.image_canvas = tk.Canvas(self)
        self.image_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.image_canvas.bind("<Configure>", self.on_resize)
        self.image_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.image_canvas.bind("<B1-Motion>", self.on_move_press)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.start_click = None
        self.rect = None
        self.canvas_image = None
        self.image = None
        self.resize_after = None

        ###########################################################################################

        self.pack(fill=tk.BOTH, expand=True)
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
        self.gpu_double_precision.set(self.compute_config.gpu_double_precision)
        self.write_zoom_entry(self.compute_config.get_zoom())
        self.write_center_entry(self.compute_config.get_center())

    def write_config(self):
        self.compute_config.max_iterations = int(self.max_iterations.get())
        self.compute_config.perturbation = self.perturbation.get()
        self.compute_config.gpu = self.gpu.get()
        self.compute_config.gpu_double_precision = self.gpu_double_precision.get()
        self.compute_config.set_image_dimensions(
            round(self.resolution_scale.get() * self.image_canvas.winfo_width()),
            round(self.resolution_scale.get() * self.image_canvas.winfo_height())
        )
        return self.set_zoom()

    def on_resize(self, event):
        if self.resize_after is not None:
            self.after_cancel(self.resize_after)
        self.resize_after = self.after(100, self.compute_and_draw)

    def set_zoom(self):
        zoom_text = self.zoom.get()
        try:
            self.compute_config.set_zoom(mpfr(zoom_text))
            self.zoom_frame.config(fg='black')
            return True
        except ValueError:
            self.zoom_frame.config(fg='red')
            return False

    def increase_zoom(self):
        self.write_zoom_entry(2 * self.compute_config.get_zoom())
        self.compute_and_draw()

    def decrease_zoom(self):
        self.write_zoom_entry(self.compute_config.get_zoom() / 2)
        self.compute_and_draw()

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
        self.zoom_frame.config(fg='black')

    def write_center_entry(self, center: mpc):
        self.center_re.set(str(center.real))
        self.center_im.set(str(center.imag))

    def on_button_press(self, event):
        if not self.computing:
            self.start_click = event

    def on_move_press(self, event):
        if self.start_click is None:
            return
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

    def on_button_release(self, _):
        if self.rect is None:
            return

        coords = self.image_canvas.coords(self.rect)
        self.image_canvas.delete(self.rect)
        self.rect = None

        new_pix_width = coords[2] - coords[0]
        canvas_center_x = self.image_canvas.winfo_width() / 2
        canvas_center_y = self.image_canvas.winfo_height() / 2
        diff = (self.start_click.x - canvas_center_x) + 1j * (canvas_center_y - self.start_click.y)
        self.start_click = None

        if new_pix_width == 0:
            return

        width_per_pix = self.compute_config.get_width() / self.image_canvas.winfo_width()
        self.compute_config.set_center(self.compute_config.get_center() + width_per_pix * diff)
        self.compute_config.set_zoom_from_width(new_pix_width * width_per_pix)

        self.write_zoom_entry(self.compute_config.get_zoom())
        self.write_center_entry(self.compute_config.get_center())
        self.compute_and_draw()

    def recompute(self):
        self.fractal.back()
        self.compute_and_draw()

    def compute_and_draw(self):
        if self.computing:
            return
        threading.Thread(target=self.threaded_compute_and_draw).start()

    def threaded_compute_and_draw(self):
        with temp_disable(self.control_panel):
            self.computing = True
            my_logger.info("-" * 80)
            start = time.time()
            try:
                if not self.write_config():
                    return
                for result in self.fractal.compute(self.compute_config):
                    self.compute_result = result
                    self.draw(self.compute_result)
            finally:
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
        shape = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
        if (image.width, image.height) != shape:
            image = image.resize(shape)
        self.image = image
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
        # make perturbation glitches appear like brot points
        iterations = iterations + (iterations == GLITCH_ITER).astype(np.int32) * self.compute_config.max_iterations
        brot_pixels = iterations == self.compute_config.max_iterations
        iterations = convert_to_fractional_counts(iterations, points)

        colours = histogram_colouring(iterations, self.palette, brot_pixels)
        self.set_image(Image.fromarray(colours, "RGB"))

    def save_image(self):
        f = asksaveasfile(
            mode="wb",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")],
            initialfile=f"mandelbrot-{time.strftime('%Y.%m.%d-%H.%M.%S')}",
            defaultextension=".png"
        )
        if f is not None:
            with f:
                self.image.save(f)


def run(config: MandelbrotConfig):
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
    )
    root.geometry("{}x{}".format(config.image_width, config.image_height))
    root.mainloop()
