import numpy as np


BROT_COLOUR = (0, 0, 0)


def histogram_colouring(iterations, palette, brot_pixels, num_cycles=2):
    # rescale to 0 so we can use the values as indices to the cumulative counts
    iterations -= np.min(iterations)

    histogram = np.histogram(iterations, np.max(iterations) + 1)[0]
    # don't let brot pixels affect colour scaling
    histogram[-1] -= np.sum(brot_pixels)
    cumulative_counts = np.cumsum(histogram)
    # rescale so the entire colour range is used (otherwise the first colour used would be offset)
    cumulative_counts = cumulative_counts - cumulative_counts[0]
    relative_cumulative_counts = cumulative_counts[iterations] / max(1, cumulative_counts[-1])

    num_colours = palette.shape[0] - 1
    indices = (num_cycles * num_colours * relative_cumulative_counts).astype(np.int32) % num_colours

    colours = (255 * palette[indices]).astype(np.uint8)
    colours[brot_pixels] = BROT_COLOUR
    return colours


def cyclic_colouring(iterations, palette, brot_pixels, num_cycles=3):
    iterations = np.sqrt(iterations)
    iter_range = np.maximum(1, np.ptp(iterations))
    a = iter_range / num_cycles
    iterations = iterations % a / a

    num_colours = len(palette) - 1
    indices = np.round(num_colours * iterations).astype(np.int32)
    colours = (255 * palette[indices]).astype(np.uint8)
    colours[brot_pixels] = BROT_COLOUR
    return colours



def generate_palette(rgb_offsets=None):
    if rgb_offsets is None:
        rgb_offsets = np.random.random(size=3)
    cols = np.linspace(0, 1, 2**11)
    a = np.column_stack([(cols + offset) * 2 * np.pi for offset in rgb_offsets])
    return 0.5 + 0.5 * np.cos(a)

