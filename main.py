import cProfile
import os
import argparse
import pstats
import sys
import math
import numpy as np
import time
import random
import threading
from tkinter import *
from PIL import Image, ImageTk

from mandelbrot import Mandelbrot
from gmpy2 import mpc, get_context

BROT_COLOUR = (0, 0, 0)
GLITCH_COLOUR = BROT_COLOUR
REF_COLOUR = (255, 0, 0)


class FractalCanvas(Canvas):
    def __init__(self, parent, height, width, t_left: mpc, b_right: mpc, iterations=None, save=False,
                 use_multiprocessing: bool = True, use_gpu: bool = False, perturbation: bool = False,
                 num_probes: int = 25,
                 num_series_terms: int = 7):
        Canvas.__init__(self, parent)
        self.pixels = None
        self.parent = parent
        self.parent.title("Mandelbrot")

        self.canvasW, self.canvasH = width, height

        self.multiprocessing = BooleanVar(value=use_multiprocessing)
        self.perturbation = BooleanVar(value=perturbation)
        self.gpu = BooleanVar(value=use_gpu)
        self.init_iterations = iterations

        self.start_click = None
        self.rect = None
        self.canvas_image = None

        ui = Frame(self)
        ui.pack(side=TOP, fill=X)
        r_controls = Frame(ui)
        r_controls.pack(side=RIGHT)
        pertubation_controls = Frame(ui)
        pertubation_controls.pack(side=RIGHT)
        l_controls = Frame(ui)
        l_controls.pack(side=LEFT)

        copy_coords = Button(l_controls, command=self.copy_coords, text="copy coords")
        copy_coords.pack(side=RIGHT)
        recompute = Button(l_controls, command=self.compute_and_draw, text="recompute")
        recompute.pack(side=RIGHT)

        Label(l_controls, text="Max iterations", height=1).pack(side=LEFT)
        self.iterations = StringVar(value=iterations)
        iter_entry = Entry(l_controls, textvariable=self.iterations, width=10)
        iter_entry.bind('<Return>', self.on_iter_submit)
        iter_entry.pack(side=LEFT)

        check_pertubations = Checkbutton(pertubation_controls, text="high precision", variable=self.perturbation,
                                         command=self.set_pertubations)

        self.check_multiprocessing = Checkbutton(r_controls, text="multiprocessing", variable=self.multiprocessing,
                                                 command=self.set_multiprocessing)

        self.check_gpu = Checkbutton(r_controls, text="gpu", variable=self.gpu, command=self.set_gpu)

        back = Button(r_controls, command=self.go_back, text="go back")
        colour = Button(r_controls, command=self.recolour, text="recolour")
        reset = Button(r_controls, command=self.reset, text="reset")
        self.mag = Text(r_controls, height=1, width=20, state=DISABLED)
        self.mag.pack(side=RIGHT)
        reset.pack(side=RIGHT)
        back.pack(side=RIGHT)
        colour.pack(side=RIGHT)
        check_pertubations.pack(side=RIGHT)
        self.check_multiprocessing.pack(side=RIGHT)
        self.check_gpu.pack(side=RIGHT)

        self.pack(fill=BOTH, expand=1)

        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_move_press)
        self.bind("<ButtonRelease-1>", self.on_button_release)

        self.fractal = Mandelbrot(t_left=t_left, b_right=b_right, iterations=iterations, width=width, height=height,
                                  multiprocessing=self.multiprocessing.get(), gpu=self.gpu.get(), perturbations=perturbation,
                                  num_probes=num_probes, num_series_terms=num_series_terms)
        self.image_stack = []
        self.computed_cur_img = True
        self.generate_palette()
        self.pixelColors = []
        self.img = None
        self.save = save
        self.update_text_fields()
        self.compute_and_draw()

    def copy_coords(self):
        self.parent.clipboard_clear()
        self.parent.clipboard_append(f'-tlr " {str(self.fractal.t_left.real)}" '
                                     f'-tli " {str(self.fractal.t_left.imag)}" '
                                     f'-bri " {str(self.fractal.b_right.imag)}" '
                                     f'-brr " {str(self.fractal.b_right.real)}" '
                                     f'-i {self.fractal.iterations}')
        if self.fractal._perturbations:
            self.parent.clipboard_append(" -p")

        if self.fractal._gpu:
            self.parent.clipboard_append(" -g")

        self.parent.update()

    def set_pertubations(self):
        self.fractal.set_perturbations(self.perturbation.get())

    def set_multiprocessing(self):
        self.fractal.multiprocessing = self.multiprocessing.get()

    def set_gpu(self):
        self.fractal.set_gpu(self.gpu.get())

    def recolour(self):
        self.generate_palette()
        if self.computed_cur_img:
            self.draw(self.pixels)
        else:
            self.compute_and_draw()

    def update_text_fields(self):
        self.iterations.set(self.fractal.iterations)

    def reset(self):
        self.fractal.reset()

        # reset variable
        if self.fractal.iterations != self.init_iterations:
            self.fractal.iterations = self.init_iterations
            self.generate_palette()

        self.compute_and_draw()

    def set_mag(self):
        self.mag.config(state=NORMAL)
        width = self.fractal.get_width()
        self.mag.replace("1.0", "1.end", f"width: {'{:.2e}'.format(width)}")
        self.mag.config(state=DISABLED)

    def go_back(self):
        if self.image_stack:
            self.background = self.image_stack.pop()
            self.computed_cur_img = False
            self.fractal.pop_corners()
            self.set_mag()

    def on_iter_submit(self, event):
        self.fractal.iterations = int(self.iterations.get())
        self.generate_palette()
        self.compute_and_draw()

    def on_button_press(self, event):
        self.start_click = (event.x, event.y)

    def on_move_press(self, event):
        # expand rectangle as you drag the mouse
        if self.rect is None:
            self.rect = self.create_rectangle(self.start_click[0], self.start_click[1], 1, 1, fill="")

        self.coords(self.rect, self.start_click[0], self.start_click[1], event.x, event.y)
        self.tag_raise(self.rect)

    def on_button_release(self, event):
        if self.rect is None:
            return

        x = self.coords(self.rect)
        t_left_coords = (min(x[0], x[2]), min(x[1], x[3]))
        b_right_coords = (max(x[0], x[2]), max(x[1], x[3]))

        self.fractal.reposition(t_left_coords=t_left_coords, b_right_coords=b_right_coords)
        self.image_stack.append(self.background)
        self.delete(self.rect)
        self.rect = None
        self.compute_and_draw()

    def compute_and_draw(self):
        threading.Thread(target=self.threaded_compute_and_draw).start()

    def threaded_compute_and_draw(self):
        print('-' * 20)
        start = time.time()
        for pixels in self.fractal.get_pixels():
            self.pixels = pixels
            self.draw(self.pixels)

        comp_time = time.time()
        print("computation took {} seconds".format(round(comp_time - start, 2)))
        self.computed_cur_img = True

    def draw(self, pixels):
        self.draw_pixels(pixels)
        self.set_mag()
        if self.canvas_image is None:
            self.canvas_image = self.create_image(0, 0, image=self.background, anchor=NW)
        else:
            self.itemconfig(self.canvas_image, image=self.background)

    def generate_palette(self):
        iterations = self.fractal.iterations
        palette_length = iterations // 2

        pairs = [
            (2 * math.pi / (rand_b * (palette_length // 2) + palette_length // 2), 256 * rand_c)
            for rand_b, rand_c in np.random.random((3, 2))
        ]

        self.palette = np.empty((iterations + 3, 3), dtype=np.uint8)
        self.palette[0] = BROT_COLOUR
        self.palette[iterations + 1] = REF_COLOUR
        self.palette[iterations + 2] = GLITCH_COLOUR

        for i, (b, c) in enumerate(pairs):
            iter_range = np.arange(1, iterations + 1) % palette_length
            temp = 128 * (np.sin(b * iter_range + c) + 1)
            self.palette[1:iterations + 1, i] = np.maximum(0, np.minimum(temp.astype(int), 255))

    def draw_pixels(self, pixels):
        img = Image.fromarray(self.palette[pixels], "RGB")
        if self.save:
            self.save_image(img)
        self.background = ImageTk.PhotoImage(img.resize((self.canvasW, self.canvasH)))

    def save_image(self, img):
        img.save("output/{}.png".format(time.strftime("%Y-%m-%d-%H:%M:%S")), "PNG", optimize=True)


def main():
    master = Tk()
    height = round(master.winfo_screenheight() * 0.6)
    width = height
    parser = argparse.ArgumentParser(description='Generate the Mandelbrot set')
    parser.add_argument('-i', '--iterations', type=int, help='The number of iterations done for each pixel.',
                        default=500)
    parser.add_argument('-tlr', '--top-left-real', type=str,
                        help='The top-left real coordinate of the area to render in str form', default="-1.5")
    parser.add_argument('-tli', '--top-left-imag', type=str,
                        help='The top-left imag coordinate of the area to render in str form', default="1.25")
    parser.add_argument('-brr', '--bottom-right-real', type=str,
                        help='The bottom-right real coordinate of the area to render in str form', default="0.5")
    parser.add_argument('-bri', '--bottom-right-imag', type=str,
                        help='The bottom-right imag coordinate of the area to render in str form', default="-1.25")
    parser.add_argument('-w', '--width', type=int, help='The width of the image.')
    parser.add_argument('-s', '--save', action='store_true', help='Save the generated image.')
    parser.add_argument('-nm', '--no-multiprocessing', action='store_false', help="Don't use multiprocessing.")
    parser.add_argument('-p', '--perturbation', action='store_true', help="Use perturbation theory for high precision "
                                                                           "computation.")
    parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU via opencl to render")
    args = parser.parse_args()

    get_context().precision = 3000

    t_left = mpc(f"({args.top_left_real} {args.top_left_imag})")
    b_right = mpc(f"({args.bottom_right_real} {args.bottom_right_imag})")
    render = FractalCanvas(parent=master, height=height, width=width, use_multiprocessing=args.no_multiprocessing,
                           t_left=t_left, b_right=b_right, iterations=args.iterations, save=args.save, perturbation=args.perturbation, use_gpu=args.gpu)

    master.geometry("{}x{}".format(render.canvasW, render.canvasH))
    master.mainloop()


if __name__ == "__main__":

    # profile = cProfile.Profile()
    # profile.runcall(main)
    # ps = pstats.Stats(profile)
    # ps.sort_stats("cumtime")
    # ps.print_stats(30)
    main()
