#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

inline bool inRange(double a, double b, int BREAKOUT_R2) {
    return a * a + b * b < BREAKOUT_R2;
}

/* Iterates through the Mandelbrot set at location x,y up to maxTimes */
__kernel void pixel64(__global int* out, const double tl_real, const double tl_imag, const double width_per_pix,
        const int maxTimes, const float s, const float p, const int width, const int BREAKOUT_R2) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int times = 0;
    double z_real = s;
    double z_imag = p;
    double c_real = tl_real + x*width_per_pix;
    double c_imag = tl_imag - y*width_per_pix;
    double temp;

    while (times < maxTimes && inRange(z_real, z_imag, BREAKOUT_R2)) {
        temp = z_real;
        z_real = z_real * z_real - z_imag * z_imag + c_real;
        z_imag = 2 * temp * z_imag + c_imag;
        times++;
    }
    if (times == maxTimes) {
        times = 0;
    }
    out[y * width + x] = times;
}



