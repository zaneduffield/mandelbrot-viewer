#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

inline bool inRange(cdouble_t c, int BREAKOUT_R2) {
    return c.real*c.real + c.imag*c.imag < BREAKOUT_R2;
}

/* Iterates through the Mandelbrot set at location x,y up to maxTimes */
__kernel void pixel64(__global int* out, const cdouble_t tl, const double width_per_pix,
        const int maxTimes, const cdouble_t z_0, const int width, const int BREAKOUT_R2) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int times = 0;
    cdouble_t z = z_0;
    cdouble_t c = cdouble_add(tl ,cdouble_new(x*width_per_pix, -y*width_per_pix));

    while (times < maxTimes && inRange(z, BREAKOUT_R2)) {
        z = cdouble_add(cdouble_mul(z, z), c);
        times++;
    }
    if (times == maxTimes) {
        times = 0;
    }
    out[y * width + x] = times;
}



