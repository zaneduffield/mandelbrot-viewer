#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

inline bool inRange(cfloat_t c, int BREAKOUT_R2) {
  return c.real * c.real + c.imag * c.imag < BREAKOUT_R2;
}

/* Iterates through the Mandelbrot set at location x,y up to maxIter */
__kernel void pixel64(
    __global int *iterations,
    __global cfloat_t *points,
    const cfloat_t tl,
    const float width_per_pix,
    const float height_per_pix,
    const int maxIter,
    const cfloat_t z_0,
    const int width,
    const int BREAKOUT_R2
){
  int y = get_global_id(0);
  int x = get_global_id(1);

  cfloat_t z = z_0;
  cfloat_t c =
      cfloat_add(tl, cfloat_new(x * width_per_pix, -y * height_per_pix));

  int i = 0;
  while (i < maxIter && inRange(z, BREAKOUT_R2)) {
    z = cfloat_add(cfloat_mul(z, z), c);
    i++;
  }
  iterations[y * width + x] = i;
  points[y * width + x] = z;
}
