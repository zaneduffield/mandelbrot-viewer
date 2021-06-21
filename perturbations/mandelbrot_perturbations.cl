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
    const double term_scaling_factor,
    const cdouble_t init_delta,
    const int iteration
 ){
    cdouble_t scaled_delta = cdouble_mulr(init_delta, term_scaling_factor);
    cdouble_t init_scaled_delta = scaled_delta;
    cdouble_t out = cdouble_new(0.0, 0.0);
    for (int k=0; k < num_terms; k++){
        out = cdouble_add(out, cdouble_mul(terms[iteration * num_terms + k], scaled_delta));
        scaled_delta = cdouble_mul(scaled_delta, init_scaled_delta);
    }
    return out;
 }

inline int search_for_escape(
    __constant cdouble_t* precise_reference,
    const int num_terms,
    __constant cdouble_t* terms,
    const double term_scaling_factor,
    const int ref_x,
    const int ref_y,
    const int w_per_pix,
    int lo,
    int hi,
    const int x,
    const int y,
    const int BREAKOUT_R2
){
    int mid;
    cdouble_t delta, point;
    while (hi != lo) {
        mid = (lo + hi)/2;
        delta = get_delta(ref_x, ref_y, w_per_pix, x, y);
        point = cdouble_add(precise_reference[mid], get_estimate(num_terms, terms, term_scaling_factor, delta, mid));
        if (sq_mod(point) <= BREAKOUT_R2){
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
    const double term_scaling_factor,
    const int breakout,
    __constant cdouble_t* precise_reference,
    const int iter_accurate,
    const int iterations,
    const int glitch_iter,
    const int BREAKOUT_R2
){
    int x = get_global_id(0);
    int y = get_global_id(1);

    cdouble_t delta_0 = get_delta(ref_x, ref_y, w_per_pix, x, y);
    cdouble_t delta_i = get_estimate(num_terms, terms, term_scaling_factor, delta_0, iter_accurate-1);
    int this_breakout = 0;
    for (int i=iter_accurate; i<breakout; i++) {
        cdouble_t x_i = precise_reference[i-1];
        cdouble_t point = cdouble_add(delta_i, x_i);
        float actual_size = sq_mod(point);

        if (actual_size < 0.00000001 * sq_mod(x_i)) {
            out[y * width + x] = glitch_iter;
            return;
        }

        if (actual_size <= BREAKOUT_R2){
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
        out[y * width + x] = search_for_escape(precise_reference, num_terms, terms, term_scaling_factor, ref_x, ref_y, w_per_pix, 0, iter_accurate, x, y, BREAKOUT_R2) + 1;
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
    const double term_scaling_factor,
    const int breakout,
    __constant cdouble_t* precise_reference,
    const int iter_accurate,
    const int iterations,
    const int glitch_iter,
    const int fix_glitches,
    const int BREAKOUT_R2
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (fix_glitches == 0 || out[y * width + x] == glitch_iter) {
        approximate_pixel(out, w_per_pix, width, ref_x, ref_y, terms, num_terms, term_scaling_factor, breakout, precise_reference, iter_accurate, iterations, glitch_iter, BREAKOUT_R2);
    }
}
