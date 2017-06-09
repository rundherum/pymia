import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_2d_segmentation(path: str, image, ground_truth, segmentation, alpha: float=0.5, label: int=1):
    """
    Plots a 2-dimensional image with an overlaid mask, which indicates under-, correct-, and over-segmentation.

    :param path: The save path.
    :type path: str
    :param image: The image.
    :type image: np.ndarray
    :param ground_truth: The ground truth.
    :type ground_truth: np.ndarray
    :param segmentation: The segmentation.
    :type segmentation: np.ndarray
    :param alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
    :type alpha: float
    :param label: The segmentation's label.
    :type label: int

    Example usage:

    >>> img = np.random.randn(10, 15) * 0.1
    >>> ground_truth = np.zeros((10, 15))
    >>> ground_truth[3:-3, 3:-3] = 1
    >>> segmentation = np.zeros((10, 15))
    >>> segmentation[4:-2, 4:-2] = 1
    >>> plotter.plot_2d_segmentation("/your/path/plot_2d_segmentation.png", img, ground_truth, segmentation)
    """

    if not image.shape == ground_truth.shape == segmentation.shape:
        raise ValueError("image, ground_truth, and segmentation must have equal shape")
    if not image.ndim == 2:
        raise ValueError("only 2-dimensional images supported")

    mask = np.zeros(ground_truth.shape)
    mask[np.bitwise_and(ground_truth == label, segmentation != label)] = 1  # under-segmentation
    mask[np.bitwise_and(ground_truth == label, segmentation == label)] = 2  # correct segmentation
    mask[np.bitwise_and(ground_truth != label, segmentation == label)] = 3  # over-segmentation
    masked = np.ma.masked_where(mask == 0, mask)

    fig = plt.figure()
    # configure axes such that no boarder is plotted
    # refer to https://github.com/matplotlib/matplotlib/issues/7940/ about how to remove axis from plot
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.margins(0)
    ax.tick_params(which='both', direction='in')

    # plot image and mask
    ax.imshow(image, 'gray', interpolation='none')
    cm = LinearSegmentedColormap.from_list('rgb', [(1, 0, 0), (0, 1, 0), (0, 0, 1)], N=3)  # simple RGB color map
    ax.imshow(masked, interpolation='none', alpha=alpha, cmap=cm)

    fig.add_axes(ax)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(path, bbox_inches=extent)
    plt.close()
