import time
import numpy as np

from numba import njit

from brot.utils.mandelbrot_utils import my_logger


def get_new_ref(blob_grid: np.ndarray):
    lo = np.array([0, 0], dtype=np.int64)
    hi = np.array(blob_grid.shape, dtype=np.int64)
    start = time.time()
    refs = _get_new_refs(blob_grid, lo, hi)
    my_logger.debug(f"found {len(refs)} new refs in {time.time() - start} seconds")
    return max(refs)[1:]


@njit()
def _get_search_size(lo, hi):
    return (hi[0] - lo[0]) * (hi[1] - lo[1])


@njit(inline="always")
def _add_new_ref(lo, hi, refs):
    size = _get_search_size(lo, hi)
    refs.append((size, (hi[0] + lo[0]) // 2, (hi[1] + lo[1]) // 2))


@njit()
def _add_new_search(lo, hi, searches, refs):
    if len(refs) == 0 or _get_search_size(lo, hi) > max(refs)[0]:
        searches.append((lo, hi))


@njit()
def _get_new_refs(
    blob_grid: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_sectors=20, max_refs=5
):
    searches = [(lo, hi)]
    refs = []
    while searches and len(refs) < max_refs:
        (lo, hi) = searches.pop()
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            _add_new_ref(lo, hi, refs)
            continue

        dim = int(hi[0] - lo[0] < hi[1] - lo[1])
        num_sectors = min(hi[dim] - lo[dim], max_sectors)
        blob_counts = np.zeros(num_sectors)
        sector_width = (hi[dim] - lo[dim]) / num_sectors
        sector = 0
        for i in range(lo[0], hi[0]):
            if not dim:
                sector = int((i - lo[0]) / sector_width)
            for j in range(lo[1], hi[1]):
                if dim:
                    sector = int((j - lo[1]) / sector_width)
                blob_counts[sector] += int(blob_grid[i, j])

        total = np.sum(blob_counts)
        if total == 0 or total / _get_search_size(lo, hi) >= 0.9:
            _add_new_ref(lo, hi, refs)
            continue

        selection = None
        index = 0
        while index <= num_sectors:
            if index < num_sectors and blob_counts[index] / total > 1 / num_sectors:
                if selection is None:
                    selection = [index, index + 1]
                else:
                    selection[-1] = index + 1
            elif selection is not None:
                new_lo = lo.copy()
                new_hi = hi.copy()
                new_lo[dim] += selection[0] * sector_width
                new_hi[dim] = lo[dim] + selection[1] * sector_width
                _add_new_search(new_lo, new_hi, searches, refs)
                selection = None

            index += 1

    return refs
