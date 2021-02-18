import cProfile
import os
import argparse
import pstats
import sys
import math
import numpy as np
import time
import random
from tkinter import *
from PIL import Image, ImageTk

from mandelbrot import Mandelbrot
from gmpy2 import mpc, get_context

GLITCH_COLOUR = (255, 255, 255)
BROT_COLOUR = (0, 0, 0)
REF_COLOUR = (255, 0, 0)


class Framework(Frame):
    def __init__(self, parent, height, width, t_left: mpc, b_right: mpc, iterations=None, save=False,
                 use_multiprocessing: bool = True, use_gpu: bool = False, perturbation: bool = False,
                 palette_length: int = 300,
                 num_probes: int = 4,
                 num_series_terms: int = 7):
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Mandelbrot")

        self.canvasW, self.canvasH = width, height
        self.palette_length = palette_length

        self.multiprocessing = BooleanVar()
        self.multiprocessing.set(use_multiprocessing)
        self.perturbation = BooleanVar()
        self.perturbation.set(perturbation)
        self.gpu = BooleanVar()
        self.gpu.set(use_gpu)
        self.init_iterations = iterations

        self.start_click = None
        self.rect = None

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
        self.canvas = Canvas(self)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.fractal = Mandelbrot(t_left=t_left, b_right=b_right, iterations=iterations, width=width, height=height,
                                  multiprocessing=self.multiprocessing.get(), gpu=self.gpu.get(), perturbations=perturbation,
                                  num_probes=num_probes, num_series_terms=num_series_terms)
        self.image_stack = []
        self.computed_cur_img = True
        self.change_palette()
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
        if self.fractal.perturbations:
            self.parent.clipboard_append(" -p")
        self.parent.update()

    def set_pertubations(self):
        self.fractal.perturbations = self.perturbation.get()

    def set_multiprocessing(self):
        self.fractal.multiprocessing = self.multiprocessing.get()

    def set_gpu(self):
        self.fractal.set_gpu(self.gpu.get())
        self.perturbation.set(False)
        self.set_pertubations()

    def recolour(self):
        self.change_palette()
        if self.computed_cur_img:
            self.draw()
        else:
            self.compute_and_draw()

    def update_text_fields(self):
        self.iterations.set(self.fractal.iterations)

    def reset(self):
        self.fractal.reset()

        # reset variable
        # TODO: fix resetting of new variables
        if self.fractal.iterations != self.init_iterations:
            self.fractal.iterations = self.init_iterations
            self.set_palette()

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
        self.set_palette()
        self.compute_and_draw()

    def on_series_submit(self, event):
        self.fractal.num_series_terms = int(self.num_terms.get())
        self.compute_and_draw()

    def on_button_press(self, event):
        self.start_click = (event.x, event.y)

    def on_move_press(self, event):
        # expand rectangle as you drag the mouse
        if self.rect is None:
            self.rect = self.canvas.create_rectangle(self.start_click[0], self.start_click[1], 1, 1, fill="")

        self.canvas.coords(self.rect, self.start_click[0], self.start_click[1], event.x, event.y)

    def on_button_release(self, event):
        if self.rect is None:
            return

        x = self.canvas.coords(self.rect)
        t_left_coords = (min(x[0], x[2]), min(x[1], x[3]))
        b_right_coords = (max(x[0], x[2]), max(x[1], x[3]))

        self.fractal.reposition(t_left_coords=t_left_coords, b_right_coords=b_right_coords)
        self.image_stack.append(self.background)
        self.canvas.delete(self.rect)
        self.rect = None
        self.compute_and_draw()

    def compute_and_draw(self):
        print('-' * 20)
        start = time.time()
        self.fractal.get_pixels()
        comp_time = time.time()
        print("computation took {} seconds".format(round(comp_time - start, 2)))
        self.draw()
        self.computed_cur_img = True

    def draw(self):
        self.draw_pixels()
        self.set_mag()
        self.canvas.create_image(0, 0, image=self.background, anchor=NW)
        self.canvas.pack(fill=BOTH, expand=1)

    def set_palette(self):
        iterations = self.fractal.iterations
        self.palette_length = iterations // 2

        redb = 2 * math.pi / (self.red_rand_b * (self.palette_length // 2) + self.palette_length // 2)
        redc = 256 * self.red_rand_c
        greenb = 2 * math.pi / (self.green_rand_b * (self.palette_length // 2) + self.palette_length // 2)
        greenc = 256 * self.green_rand_c
        blueb = 2 * math.pi / (self.blue_rand_b * (self.palette_length // 2) + self.palette_length // 2)
        bluec = 256 * self.blue_rand_c

        self.palette = np.empty((iterations + 3, 3), dtype=np.uint8)
        self.palette[0] = BROT_COLOUR
        for i in range(1, iterations + 1):
            x = i % self.palette_length
            r = clamp(int(128 * (math.sin(redb * x + redc) + 1)))
            g = clamp(int(128 * (math.sin(greenb * x + greenc) + 1)))
            b = clamp(int(128 * (math.sin(blueb * x + bluec) + 1)))
            self.palette[i] = [r, g, b]
        self.palette[iterations + 1] = REF_COLOUR
        self.palette[iterations + 2] = GLITCH_COLOUR

    def generate_palette_variables(self):
        self.red_rand_b = random.random()
        self.red_rand_c = random.random()
        self.green_rand_b = random.random()
        self.green_rand_c = random.random()
        self.blue_rand_b = random.random()
        self.blue_rand_c = random.random()

    def change_palette(self):
        self.generate_palette_variables()
        self.set_palette()

    def draw_pixels(self):
        img = Image.fromarray(self.palette[self.fractal.pixels], "RGB")
        self.img = img
        if self.save:
            self.save_image(None)
        photoimg = ImageTk.PhotoImage(img.resize((self.canvasW, self.canvasH)))
        self.background = photoimg

    def save_image(self, event):
        self.img.save("output/{}.png".format(time.strftime("%Y-%m-%d-%H:%M:%S")), "PNG", optimize=True)


def clamp(x):
    return max(0, min(x, 255))


def main():
    master = Tk()
    height = round(master.winfo_screenheight() * 0.6)
    width = round(master.winfo_screenwidth() * 0.7)
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
    args = parser.parse_args()

    get_context().precision =100

    t_left = mpc(f"({args.top_left_real} {args.top_left_imag})")
    b_right = mpc(f"({args.bottom_right_real} {args.bottom_right_imag})")
    render = Framework(parent=master, height=height, width=width, use_multiprocessing=args.no_multiprocessing,
                       t_left=t_left, b_right=b_right, iterations=args.iterations, save=args.save, perturbation=args.perturbation)

    master.geometry("{}x{}".format(render.canvasW, render.canvasH))
    master.mainloop()


if __name__ == "__main__":

    # profile = cProfile.Profile()
    # profile.runcall(main)
    # ps = pstats.Stats(profile)
    # ps.sort_stats("cumtime")
    # ps.print_stats(30)
    main()
