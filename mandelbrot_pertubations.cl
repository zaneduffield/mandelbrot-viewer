#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable


@njit
def search_for_escape(data, lo, hi, x, y):
    while hi != lo:
        mid = (lo + hi) // 2
        point = data.precise_reference[mid] + get_estimate(data, x, y, mid)
        if point.real*point.real + point.imag*point.imag <= BREAKOUT_R:
            lo = mid + 1
        else:
            hi = mid
    return hi


@njit
def get_delta(data, i, j):
    hor_delta = (i - data.ref_coords[0]) * data.w_per_pix
    ver_delta = (data.ref_coords[1] - j) * data.h_per_pix
    return complex(hor_delta, ver_delta)


@njit
def get_estimate(data, i, j, iteration):
    init_delta = get_delta(data, i, j)
    delta = init_delta
    out = 0
    for k in range(data.num_terms):
        term = data.terms[k][iteration]
        out += term * delta
        delta *= init_delta
    return out


@njit
def approximate_pixel(data, x, y, iterations_grid):
    # TODO: why does using iter_accurate without -1 cause issues???
    return 0


__kernel void approximate_pixels(
    __global ushort* out,
    const double tl_real,
    const double tl_imag,
    const double width_per_pix,
    const int maxTimes,
    const float s,
    const float p,
    const int width,
    __global int* ref_coords,
    __global complex** terms,
    const int num_terms,
    const int breakout,
    __global complex* precise_reference,
    const int iter_accurate,
    const int iterations,
    const int glitch_iter,
    const bool fix_glitches
){

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (!fix_glitches || out[y * width + x] == glitch_iter) {
        complex delta_i = get_estimate(data, x, y, iter_accurate-1);
        complex delta_0 = get_delta(data, x, y);
        int this_breakout = 0;
        for (int i=iter_accurate; i<breakout; i++) {
            complex x_i = precise_reference[i-1]
            complex point = delta_i + x_i;
            float actual_size = point.real * point.real + point.imag * point.imag;

            if (actual_size < 0.000001 * (x_i.real*x_i.real + x_i.imag*x_i.imag)) {
                out[y * width + x] = glitch_iter;
                return;
            }

            if (actual_size <= BREAKOUT_R){
                delta_i = 2 * precise_reference[i-1] * delta_i + delta_i * delta_i + delta_0
            } else {
                break;
            }
            this_breakout = i + 1;
        }

        if (this_breakout == 0){
            // broke out before iterating, find true breakout value using binary search on accurate estimations
            out[y * width + x] = search_for_escape(data, 0, iter_accurate, x, y) + 1;
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
}
