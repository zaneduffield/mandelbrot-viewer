import numpy as np


BROT_COLOUR = (0, 0, 0)

RGB_OFFSET_PRESETS = (
    (0.28, 0.422, 0.438),
    (0.0358, 0.984, 0.927),
    (0.522, 0.697, 0.677),
    (0.771, 0.645, 0.702),
    (0.228, 0.422, 0.356),
    (0.173, 0.204, 0.955),
    (0.933, 0.0559, 0.0138),
    (0.325, 0.166, 0.571),
    (0.576, 0.668, 0.00772),
    (0.621, 0.769, 0.987),
)
CURR_PALETTE = 0


def histogram_colouring(iterations, palette, brot_pixels, num_cycles=2, scale=100):
    # rescale to 0 so we can use the values as indices to the cumulative counts
    iterations -= np.min(iterations)
    iterations = (scale * iterations).astype(np.int64)

    histogram = np.histogram(iterations, np.max(iterations) + 1)[0]
    # don't let brot pixels affect colour scaling
    histogram[-1] -= np.sum(brot_pixels)
    cumulative_counts = np.cumsum(histogram)
    # rescale so the entire colour range is used (otherwise the first colour used would be offset)
    cumulative_counts = cumulative_counts - cumulative_counts[0]
    relative_cumulative_counts = cumulative_counts[iterations] / max(
        1, cumulative_counts[-1]
    )

    num_colours = palette.shape[0] - 1
    indices = (num_cycles * num_colours * relative_cumulative_counts).astype(
        np.int32
    ) % num_colours

    colours = (255 * palette[indices]).astype(np.uint8)
    colours[brot_pixels] = BROT_COLOUR
    return colours


def cyclic_colouring(
    iterations,
    palette,
    brot_pixels,
    num_cycles=3,
    iter_range=10,
):
    iterations = np.sqrt(iterations)
    a = iter_range / num_cycles
    iterations = iterations % a / a

    num_colours = len(palette) - 1
    indices = np.round(num_colours * iterations).astype(np.int32)
    colours = (255 * palette[indices]).astype(np.uint8)
    colours[brot_pixels] = BROT_COLOUR
    return colours


def next_palette():
    global CURR_PALETTE
    out = generate_palette(RGB_OFFSET_PRESETS[CURR_PALETTE % len(RGB_OFFSET_PRESETS)])
    CURR_PALETTE += 1
    return out


def generate_palette(rgb_offsets=None):
    if rgb_offsets is None:
        rgb_offsets = np.random.random(size=3)
    cols = np.linspace(0, 1, 2 ** 11)
    a = np.column_stack([(cols + offset) * 2 * np.pi for offset in rgb_offsets])
    return 0.5 + 0.5 * np.cos(a)
