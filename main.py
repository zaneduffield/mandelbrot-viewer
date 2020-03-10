import os
import argparse
import sys
import math
import numpy as np
import time
import random
from tkinter import *
from PIL import Image, ImageTk
from numba import njit

from complex_bf import ComplexBf
from mandelbrot import Mandelbrot
from bigfloat import BigFloat, Context, setcontext

GLITCH_COLOUR = (255, 255, 255)
BROT_COLOUR = (0, 0, 0)
REF_COLOUR = (255, 0, 0)


class Framework(Frame):
    def __init__(self, parent, height, width, t_left: ComplexBf, b_right: ComplexBf, iterations=None, save=False,
                 use_multiprocessing: bool = True, pertubations: bool = False, palette_length: int = 300,
                 num_probes: int = 100,
                 num_series_terms: int = 5):
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Mandelbrot")

        self.canvasW, self.canvasH = width, height
        self.palette_length = palette_length

        self.multiprocessing = BooleanVar()
        self.multiprocessing.set(use_multiprocessing)
        self.pertubations = BooleanVar()
        self.pertubations.set(pertubations)
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

        check_pertubations = Checkbutton(pertubation_controls, text="use pertubations", variable=self.pertubations,
                                         command=self.set_pertubations)

        Label(pertubation_controls, text="Num probes", height=1).pack(side=LEFT)
        self.num_probes = StringVar(value=num_probes)
        probes_entry = Entry(pertubation_controls, textvariable=self.num_probes, width=4)
        probes_entry.bind('<Return>', self.on_probes_submit)
        probes_entry.pack(side=LEFT)

        Label(pertubation_controls, text="Num series terms", height=1).pack(side=LEFT)
        self.num_terms = StringVar(value=num_series_terms)
        series_entry = Entry(pertubation_controls, textvariable=self.num_terms, width=3)
        series_entry.bind('<Return>', self.on_series_submit)
        series_entry.pack(side=LEFT)

        self.check_multiprocessing = Checkbutton(r_controls, text="use multiprocessing", variable=self.multiprocessing,
                                                 command=self.set_multiprocessing)
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

        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.fractal = Mandelbrot(t_left=t_left, b_right=b_right, iterations=iterations, width=width, height=height,
                                  multiprocessing=self.multiprocessing.get(), pertubations=pertubations,
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
        self.parent.clipboard_append(f'-tlr " {float(self.fractal.t_left.real)}" '
                                     f'-tli " {float(self.fractal.t_left.imag)}" '
                                     f'-bri " {float(self.fractal.b_right.imag)}" '
                                     f'-brr " {float(self.fractal.b_right.real)}" '
                                     f'-i {self.fractal.iterations}')
        self.parent.update()

    def set_pertubations(self):
        self.fractal.pertubations = self.pertubations.get()

    def set_multiprocessing(self):
        self.fractal.multiprocessing = self.multiprocessing.get()

    def recolour(self):
        self.change_palette()
        if self.computed_cur_img:
            self.draw()
        else:
            self.compute_and_draw()

    def update_text_fields(self):
        self.iterations.set(self.fractal.iterations)
        self.num_terms.set(self.fractal.num_series_terms)
        self.num_probes.set(self.fractal.num_probes)

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

    def on_probes_submit(self, event):
        self.fractal.num_probes = int(self.num_probes.get())
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
        self.palette_length = iterations // 4

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
    args = parser.parse_args()

    setcontext(Context(precision=200))

    t_left = ComplexBf(BigFloat(args.top_left_real), BigFloat(args.top_left_imag))
    b_right = ComplexBf(BigFloat(args.bottom_right_real), BigFloat(args.bottom_right_imag))
    print(t_left, b_right)
    render = Framework(parent=master, height=height, width=width, use_multiprocessing=args.no_multiprocessing,
                       t_left=t_left, b_right=b_right, iterations=args.iterations, save=args.save)

    master.geometry("{}x{}".format(render.canvasW, render.canvasH))
    master.mainloop()


if __name__ == "__main__":
    main()
