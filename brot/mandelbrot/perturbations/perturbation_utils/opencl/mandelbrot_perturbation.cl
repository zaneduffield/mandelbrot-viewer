#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

inline float sq_mod(cfloat_t c){
    return c.real*c.real + c.imag*c.imag;
}

inline cfloat_t get_init_delta(
    const int ref_x,
    const int ref_y,
    const float w_per_pix,
    const float h_per_pix,
    const int x,
    const int y
){
    return cfloat_new((x - ref_x)* w_per_pix, (ref_y - y) * h_per_pix);
}

inline cfloat_t get_delta_estimate(
    const int num_terms,
    __constant cfloat_t* terms,
    const float term_scaling_factor,
    const cfloat_t init_delta,
    const int iteration
 ){
    cfloat_t scaled_delta = cfloat_mulr(init_delta, term_scaling_factor);
    cfloat_t init_scaled_delta = scaled_delta;
    cfloat_t out = cfloat_new(0.0, 0.0);
    int term_index_offset = (iteration - 1) * num_terms;
    for (int k=0; k < num_terms; k++){
        out = cfloat_add(out, cfloat_mul(terms[term_index_offset + k], scaled_delta));
        scaled_delta = cfloat_mul(scaled_delta, init_scaled_delta);
    }
    return out;
 }

 inline cfloat_t get_reference(__constant cfloat_t* precise_reference, int i) {
     return precise_reference[i - 1];
 }

__kernel void approximate_pixels(
    __global int* iterations,
    __global cfloat_t* points,
    const float w_per_pix,
    const float h_per_pix,
    const int width,
    const int ref_x,
    const int ref_y,
    __constant cfloat_t* terms,
    const int num_terms,
    const float term_scaling_factor,
    const int breakout,
    __constant cfloat_t* precise_reference,
    const int iter_accurate,
    const int max_iterations,
    const int GLITCH_ITER,
    const float GLITCH_DIFF_THRESHOLD,
    const int fix_glitches,
    const int BREAKOUT_R2
){
    int y = get_global_id(0);
    int x = get_global_id(1);
    const int index = y * width + x;
    if (fix_glitches != 0 && iterations[index] != GLITCH_ITER) {
        return;
    }

    const cfloat_t delta_0 = get_init_delta(ref_x, ref_y, w_per_pix, h_per_pix, x, y);
    cfloat_t delta_i = get_delta_estimate(num_terms, terms, term_scaling_factor, delta_0, iter_accurate);
    cfloat_t ref_i, point = cfloat_new(0, 0);
    float point_sq_mod;
    int i = iter_accurate;
    while (i <= breakout) {
        ref_i = get_reference(precise_reference, i);
        point = cfloat_add(delta_i, ref_i);
        point_sq_mod = sq_mod(point);

        if (point_sq_mod > BREAKOUT_R2) {
            break;
        }

        if (point_sq_mod < GLITCH_DIFF_THRESHOLD * sq_mod(ref_i) ||
                (i == breakout && breakout < max_iterations)) {
            iterations[index] = GLITCH_ITER;
            return;
        }

        delta_i = cfloat_add(
            cfloat_rmul(2, cfloat_mul(get_reference(precise_reference, i), delta_i)),
            cfloat_add(
                cfloat_mul(delta_i, delta_i),
                delta_0
            )
        );
        i += 1;
    }

    if (i == iter_accurate && i <= breakout){
        // broke out before iterating, find true breakout value using binary search on accurate estimations
        int mid, lo = 1, hi = i - 1;
        while (hi != lo) {
            mid = (lo + hi)/2;
            point = cfloat_add(
                get_reference(precise_reference, mid), 
                get_delta_estimate(num_terms, terms, term_scaling_factor, delta_0, mid)
            );
            if (sq_mod(point) <= BREAKOUT_R2){
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        point = cfloat_add(
            get_reference(precise_reference, hi), 
            get_delta_estimate(num_terms, terms, term_scaling_factor, delta_0, hi)
        );
        i = hi;
    }

    iterations[index] = i - 1;
    points[index] = point;
}
