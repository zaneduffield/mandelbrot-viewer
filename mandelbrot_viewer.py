import argparse

from gmpy2 import mpfr, mpc, get_context

from ui import tkinter_ui
from utils.mandelbrot_utils import MandelbrotConfig, my_logger


def make_cli_args(config: MandelbrotConfig):
    center = (config.b_right + config.t_left) / 2
    zoom = 2 / (config.b_right.real - config.t_left.real)
    args = (
        f'--center "{str(center.real)} {str(center.imag)}" -z {zoom}'
        f' -i {config.iterations} --height {config.height} --width {config.width}'
    )

    if config.perturbation:
        args += " -p"

    if config.gpu:
        args += " -g"

    return args


def main():
    parser = argparse.ArgumentParser(description="Generate the Mandelbrot set")
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="The number of iterations done for each pixel.",
        default=500,
    )
    parser.add_argument(
        "-c",
        "--center",
        type=str,
        help="the complex number center to start at in '<real> <imag>' format",
        default="-0.5 0",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        type=str,
        help="zoom",
        default="0.5E+0",
    )
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)

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
        help="Use perturbation theory for high precision computation.",
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Use GPU via opencl to render"
    )
    parser.add_argument("-log", "--log-level", choices=["debug", "info", "warning"], default="debug")

    args = parser.parse_args()

    my_logger.setLevel(args.log_level.upper())

    get_context().precision = int(len(args.center) * 3.32)  # log2(10) ~= 3.32
    width = 2 / mpfr(args.zoom)
    center = mpc(args.center)

    half_diag = (width / 2) * (-1 + 1j)
    t_left = center + half_diag
    b_right = center - half_diag

    config = MandelbrotConfig(
        width=max(args.width, 0),
        height=max(args.height, 0),
        t_left=t_left,
        b_right=b_right,
        iterations=args.iterations,
        perturbation=args.perturbation,
        gpu=args.gpu,
        num_series_terms=10,
    )

    tkinter_ui.run(config, args.save)


if __name__ == "__main__":
    main()
