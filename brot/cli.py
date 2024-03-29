import argparse

from gmpy2 import mpfr, mpc, get_context
from brot.utils.constants import GLITCH_ERROR_THRESHOLD

from brot.utils.mandelbrot_utils import MandelbrotConfig, my_logger


def make_cli_args(config: MandelbrotConfig):
    center = (config.b_right() + config.t_left()) / 2
    zoom = 2 / (config.b_right().real - config.t_left().real)
    args = (
        f'--center "{str(center.real)} {str(center.imag)}" -z {zoom:.4e}'
        f" -i {config.max_iterations} --height {config.image_height} --width {config.image_width}"
    )

    if config.perturbation:
        args += " -p"

    if config.gpu:
        args += " -g"

    if config.gpu_double_precision:
        args += " -gd"

    return args


def make_config():
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
        default="1E+0",
    )
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument(
        "-p",
        "--perturbation",
        action="store_true",
        help="Use perturbation theory for high precision computation.",
    )
    parser.add_argument(
        "-gt", "--glitch-error-threshold", type=float, default=GLITCH_ERROR_THRESHOLD
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Use GPU via OpenCL to render"
    )
    parser.add_argument(
        "-gd",
        "--gpu-double",
        action="store_true",
        help="Use double precision in OpenCL",
    )
    parser.add_argument(
        "-log", "--log-level", choices=["debug", "info", "warning"], default="debug"
    )

    args = parser.parse_args()

    my_logger.setLevel(args.log_level.upper())

    get_context().precision = int(len(args.center) * 3.32)  # log2(10) ~= 3.32

    config = MandelbrotConfig(
        image_width=max(args.width, 0),
        image_height=max(args.height, 0),
        max_iterations=args.iterations,
        perturbation=args.perturbation,
        gpu=args.gpu,
        gpu_double_precision=args.gpu_double,
        glitch_error_threshold=args.glitch_error_threshold,
    )
    config.set_center_and_zoom(center=mpc(args.center), zoom=mpfr(args.zoom))

    return config
