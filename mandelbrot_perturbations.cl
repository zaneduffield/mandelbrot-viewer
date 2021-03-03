#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>


inline float sq_mod(cdouble_t c){
    return c.real*c.real + c.imag*c.imag;
}

inline cdouble_t get_delta(
    const int ref_x,
    const int ref_y,
    const double w_per_pix,
    const int x,
    const int y
){
    return cdouble_new((x - ref_x)* w_per_pix, (ref_y - y) * w_per_pix);
}

inline cdouble_t get_estimate(
    const int num_terms,
    __constant cdouble_t* terms,
    const cdouble_t init_delta,
    const int iteration
 ){
    cdouble_t delta = init_delta;
    cdouble_t out = cdouble_new(0.0, 0.0);
    for (int k=0; k < num_terms; k++){
        out = cdouble_add(out, cdouble_mul(terms[iteration * num_terms + k], delta));
        delta = cdouble_mul(delta, init_delta);
    }
    return out;
 }

inline int search_for_escape(
    __constant cdouble_t* precise_reference,
    const int num_terms,
    __constant cdouble_t* terms,
    const int ref_x,
    const int ref_y,
    const int w_per_pix,
    int lo,
    int hi,
    const int x,
    const int y
){
    int mid;
    cdouble_t delta, point;
    while (hi != lo) {
        mid = (lo + hi)/2;
        delta = get_delta(ref_x, ref_y, w_per_pix, x, y);
        point = cdouble_add(precise_reference[mid], get_estimate(num_terms, terms, delta, mid));
        if (sq_mod(point) <= 10){
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return hi;
}

void approximate_pixel(
    __global int* out,
    const double w_per_pix,
    const int width,
    const int ref_x,
    const int ref_y,
    __constant cdouble_t* terms,
    const int num_terms,
    const int breakout,
    __constant cdouble_t* precise_reference,
    const int iter_accurate,
    const int iterations,
    const int glitch_iter
){
    int x = get_global_id(0);
    int y = get_global_id(1);

    cdouble_t delta_0 = get_delta(ref_x, ref_y, w_per_pix, x, y);
    cdouble_t delta_i = get_estimate(num_terms, terms, delta_0, iter_accurate-1);
    int this_breakout = 0;
    for (int i=iter_accurate; i<breakout; i++) {
        cdouble_t x_i = precise_reference[i-1];
        cdouble_t point = cdouble_add(delta_i, x_i);
        // float actual_size = point.real * point.real + point.imag * point.imag;
        float actual_size = sq_mod(point);

        if (actual_size < 0.000001 * sq_mod(x_i)) {
            out[y * width + x] = glitch_iter;
            return;
        }

        if (actual_size <= 10){
            delta_i = cdouble_add(
                cdouble_mulr(cdouble_mul(precise_reference[i-1], delta_i), 2),
                cdouble_add(cdouble_mul(delta_i, delta_i), delta_0)
            );
        } else {
            break;
        }
        this_breakout = i + 1;
    }

    if (this_breakout == 0){
        // broke out before iterating, find true breakout value using binary search on accurate estimations
        out[y * width + x] = search_for_escape(precise_reference, num_terms, terms, ref_x, ref_y, w_per_pix, 0, iter_accurate, x, y) + 1;
    } else if (this_breakout == breakout){
        if (breakout < iterations){
            out[y * width + x] = glitch_iter;
        } else {
            out[y * width + x] = 0;
        }
    } else {
        out[y * width + x] = this_breakout;
    }
}

__kernel void approximate_pixels(
    __global int* out,
    const double w_per_pix,
    const int width,
    const int ref_x,
    const int ref_y,
    __constant cdouble_t* terms,
    const int num_terms,
    const int breakout,
    __constant cdouble_t* precise_reference,
    const int iter_accurate,
    const int iterations,
    const int glitch_iter,
    const int fix_glitches
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (fix_glitches == 0 || out[y * width + x] == glitch_iter) {
        approximate_pixel(out, w_per_pix, width, ref_x, ref_y, terms, num_terms, breakout, precise_reference, iter_accurate, iterations, glitch_iter);
    }
}
