## Mandelbrot Viewer
A performant GUI mandelbrot explorer that enables zooming beyond hardware floating point precision using perturbation theory.

## Features
- [x] CPU multiprocessing with jit compilation
- [x] GPU processing via OpenCL (double precision optional)
- [x] Deep zoom capability up to e300 on either CPU or GPU
- [x] GUI for zooming
 
## Usage
    python -m brot 

For command line interface run

    python -m brot --help

Repositioning and zooming is performed with the mouse, by clicking and dragging a rectangle around the
area you wish to zoom to.

All command line options have equivalent settings in the UI. 

## Tips

- Increase 'Max Iterations' when the boundary of the mandelbrot set becomes less detailed
- Enable 'high precision' if the image becomes pixelated, this works in CPU and GPU mode
- Enable 'gpu double precision' if the image becomes pixelated in GPU mode. This may be much slower on your GPU
- Render at a lower 'resolution scale' for better performance


## Installation

From project root

    pip install .

Optionally, if you would like GPU acceleration using OpenCL

    pip install .[gpu] 

Note: this will require having an existing OpenCL installation on your system
which you can likely get from your graphics card manufacturer.

GMPY2 and PyOpenCL can be a difficult to build on later versions of python on windows, unofficial prebuilt
binaries for both can be found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and installed with

    pip install path/to/binary.whl

after which you can repeat the original install.

## Limitations
- The max iterations must be set manually and increased as you zoom. Ideally this would be automatic.
- The colouring algorithm is relatively simple, and sometimes doesn't give the best looking images. Distance estimation and stripe-average colouring methods could be used to improve the images produced.
- For slow renders there is no user feedback on how the render is progressing. Ideally there would be
some feedback in the UI, as well as a more progressive rendering technique which initially renders in
low resolutions for faster feedback.
- When computing using perturbations the estimate on when the series approximation becomes 
innaccurate is sometimes too conservative, missing out on the large performance gains it can provide.
- The GUI experience could be smoother, especially when the renders are fast, by allowing the user to
pan and scroll instead of dragging and selecting a rectangle.
- The host memory usage of OpenCL is abnormally high when using a high number of iterations which
becomes a serious bottleneck.

