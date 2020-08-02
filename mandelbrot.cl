#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#include <pyopencl-complex.h>

inline bool inRange(float a, float b) {
    return a * a + b * b < 4.0;
}

/* Iterates through the Mandelbrot set at location x,y up to maxTimes */
__kernel void pixel64(__global int* out, const double centerX, const double centerY, const float zoom, const int maxTimes, const float s, const float p, const int sizeX, const int sizeY) {
    // Mandelbrot set is defined as:
    // f(z) = z^2 + c
    // evaluate z = f(z) until abs(z) > 2
	// return numIterations
    // Do this for our current point
    int x = get_global_id(0);
    int y = get_global_id(1);

    int times = 0;

    double a = s;
    double b = p;
    double c = (x - sizeX / 2.0) / zoom + centerX;
    double d = -(y - sizeY / 2.0) / zoom + centerY;

    while (times < maxTimes && inRange(a, b)) {
        double tmpa = (a*a) - (b*b);
        b = 2*a*b;
        a = tmpa;
        a += c;
        b += d;
        times++;
    }
    if (times == maxTimes) {
        times = 0;
    }
    out[y * sizeX + x] = times;
}



