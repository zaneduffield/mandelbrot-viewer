## Mandelbrot Viewer
A performant GUI mandelbrot explorer that enables zooming beyond hardware floating point precision using 

### Features
- [x] CPU multiprocessing with jit compilation
- [x] GPU processing via OpenCL
- [x] Deep zoom capability via perturbation theory
- [x] Deep zoom on either CPU or GPU
- [x] GUI for zooming
 
### Usage

    python -m mandelbrot-viewer
command line interface

### Limitations
- The colouring algorithm is relatively simple, and sometimes results in pretty poor looking images.
- When computing using perturbations the glitch detection is often not sensitive enough, resulting in highly distorted
images in some regions.


### Installation

Required packages can be found in requirements.txt

    pip install -r requirements.txt
    
PyOpenCL can be a little tricky to get installed on later versions of python on windows, unofficial prebuilt binaries for windows can be found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl)