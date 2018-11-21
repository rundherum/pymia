# see also https://gist.github.com/kukuruza/03731dc494603ceab0c5
import numpy as np


def get_grid_size(kernel_shape: list) -> tuple:
    size = int(np.ceil(np.sqrt(kernel_shape[-1])))
    space = 1  # the space between different kernels (in pixel)
    patch_h = kernel_shape[0]
    patch_w = kernel_shape[1]

    if patch_h == 1 and patch_w == 1:
        # 1-D conv
        w = kernel_shape[2]
        h = kernel_shape[3] + (kernel_shape[3] - 1) * space
    else:
        w = size * patch_w + (size - 1) * space
        h = size * patch_h + (size - 1) * space

    return h, w


def make_grid(kernels: np.ndarray) -> np.ndarray:
    """
    Arrange kernel weights on a grid for visualization
    :param kernels: N kernel (weights) as [N, h, w, 1]
    :return: 2D ndarray (gray-scale image)
    """
    size = int(np.ceil(np.sqrt(kernels.shape[0])))
    space = 1
    patch_h = kernels.shape[1]
    patch_w = kernels.shape[2]

    w = size * patch_w + (size-1) * space
    h = size * patch_h + (size - 1) * space

    grid = np.zeros([h, w])
    idx = 0
    for row in range(size):
        y = row * (patch_h + space)
        for col in range(size):
            x = col * (patch_w + space)
            if idx < kernels.shape[0]:
                patch = kernels[idx, :, :, 0]

                # normalize patch
                patch = (patch - np.max(patch)) / -np.ptp(patch)

                grid[y:y + patch_h, x:x + patch_w] = patch
                idx = idx + 1

    return grid


def make_grid_for_1d_conv(kernels: np.ndarray) -> np.ndarray:
    """Arrange the kernel weights on a grid for visualization purposes.

    Args:
        kernels: The kernel weights of shape (N, 1, 1, IN),
            where N is the number of 2-D convolution kernels with kernel_size=(1, 1) and
            IN is the number of input channels to the 2-D convolution.

    Returns:
        The kernel weights as array for visualization. The array has shape (N * 2 - 1, IN),
            i.e. each second row is a kernel and the other rows are black for separation.
    """
    space = 1
    w = kernels.shape[-1]
    h = kernels.shape[0] + (kernels.shape[0] - 1) * space

    grid = np.zeros([h, w])
    grid_idx = 0
    for kernel_idx in range(kernels.shape[0]):
        grid[grid_idx, :] = kernels[kernel_idx, ...]
        grid_idx += 2

    return grid
