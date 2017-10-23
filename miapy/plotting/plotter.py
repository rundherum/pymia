"""Enables the plotting of images for presentation and documentation purposes using the ``matplotlib`` library.

Refer also to `SimpleITK Notebooks
<http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/10_matplotlib's_imshow.html>`_."""
import os

import matplotlib.colors as plt_colors
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def plot_histogram(path: str, image: sitk.Image, no_bins: int=255, slice_no: int=-1,
                   title: str='', xlabel: str='', ylabel: str='') -> None:
    """Plots a histogram of an image.

    Plots either the histogram of a slice of the image or of the whole image.
    Args:
        path (str): The file path.
        image (SimpleITK.Image): The image.
        no_bins (int): The number of histogram bins.
        slice_no (int): The slice number or -1 to take the whole image.
        title (str): The histogram's title.
        xlabel (str): The histogram's x-axis label.
        ylabel (str): The histogram's y-axis label.
    """
    if slice_no > -1:
        data = sitk.GetArrayFromImage(image[:, :, slice_no])
    else:
        data = sitk.GetArrayFromImage(image)

    data = data.flatten()

    plt.hist(data, bins=no_bins)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()


def plot_histogram_overlay(path: str, image1: sitk.Image, image2: sitk.Image,
                           legend1: str='', legend2: str='',
                           no_bins: int=255, slice_no: int=-1,
                           title: str='', xlabel: str='', ylabel: str='',
                           xaxis_log: bool=False, yaxis_log: bool=False) -> None:
    """Plots a histogram of an image.

    Plots either the histogram of a slice of the image or of the whole image.
    
    Args:
        path (str): The file path.
        image1 (sitk.Image): The first image.
        image2 (sitk.Image): The second image.
        legend1 (str): The legend for the first image.
        legend2 (str): The legend for the second image.2
        no_bins (int): The number of histogram bins.
        slice_no (int): The slice number or -1 to take the whole image.
        title (str): The histogram's title.
        xlabel (str): The histogram's x-axis label.
        ylabel (str): The histogram's y-axis label.
        xaxis_log (bool): Logarithmic scale for x-axis.
        yaxis_log (bool): Logarithmic scale for y-axis.
    """
    if slice_no > -1:
        data1 = sitk.GetArrayFromImage(image1[:, :, slice_no])
        data2 = sitk.GetArrayFromImage(image2[:, :, slice_no])
    else:
        data1 = sitk.GetArrayFromImage(image1)
        data2 = sitk.GetArrayFromImage(image2)

    data1 = data1.flatten()
    data2 = data2.flatten()

    plt.hist(data1, bins=no_bins, alpha=0.5, label=legend1)
    plt.hist(data2, bins=no_bins, alpha=0.5, label=legend2)
    if legend1 or legend2: plt.legend()
    if xaxis_log: plt.xscale('log')
    if yaxis_log: plt.yscale('log')
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()


def plot_slice(path: str, image: sitk.Image, slice_no: int) -> None:
    """Plots a slice from a 3-D image to a file.

    Args:
        path (str): The output file path.
        image (sitk.Image): The 3-D image.
        slice_no (int): The slice number.
    """

    slice_ = sitk.GetArrayFromImage(image[:, :, slice_no])

    fig = plt.figure()
    # configure axes such that no boarder is plotted
    # refer to https://github.com/matplotlib/matplotlib/issues/7940/ about how to remove axis from plot
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.margins(0)
    ax.tick_params(which='both', direction='in')

    # plot image
    ax.imshow(slice_, 'gray', interpolation='none')

    fig.add_axes(ax)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(path, bbox_inches=extent)
    plt.close()


def plot_2d_segmentation(path: str,
                         image: np.ndarray,
                         ground_truth : np.ndarray,
                         segmentation : np.ndarray,
                         alpha: float=0.5,
                         label: int=1) -> None:
    """Plots a 2-dimensional image with an overlaid mask, which indicates under-, correct-, and over-segmentation.

    Args:
        path (str): The output file path.
        image (np.ndarray): The 2-D image.
        ground_truth (np.ndarray): The 2-D ground truth.
        segmentation (np.ndarray): The 2-D segmentation.
        alpha (float): The alpha blending value, between 0 (transparent) and 1 (opaque).
        label (int): The ground truth and segmentation label.

    Examples:
        >>> img = np.random.randn(10, 15) * 0.1
        >>> ground_truth = np.zeros((10, 15))
        >>> ground_truth[3:-3, 3:-3] = 1
        >>> segmentation = np.zeros((10, 15))
        >>> segmentation[4:-2, 4:-2] = 1
        >>> plot_2d_segmentation('/your/path/plot_2d_segmentation.png', img, ground_truth, segmentation)
    """

    if not image.shape == ground_truth.shape == segmentation.shape:
        raise ValueError('image, ground_truth, and segmentation must have equal shape')
    if not image.ndim == 2:
        raise ValueError('only 2-dimensional images supported')

    mask = np.zeros(ground_truth.shape)
    mask[np.bitwise_and(ground_truth == label, segmentation != label)] = 1  # under-segmentation
    mask[np.bitwise_and(ground_truth == label, segmentation == label)] = 2  # correct segmentation
    mask[np.bitwise_and(ground_truth != label, segmentation == label)] = 3  # over-segmentation
    masked = np.ma.masked_where(mask == 0, mask)

    fig = plt.figure(figsize=image.shape[::-1], dpi=2)  # figure is twice as large as array (in pixels)
    # configure axes such that no boarder is plotted
    # refer to https://github.com/matplotlib/matplotlib/issues/7940/ about how to remove axis from plot
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.margins(0)
    ax.tick_params(which='both', direction='in')

    # plot image and mask
    ax.imshow(image, 'gray', interpolation='none')
    cm = plt_colors.LinearSegmentedColormap.from_list('rgb',
                                                      [(1, 0, 0), (0, 1, 0), (0, 0, 1)], N=3)  # simple RGB color map
    ax.imshow(masked, interpolation='none', alpha=alpha, cmap=cm)

    fig.add_axes(ax)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(path, bbox_inches=extent)
    plt.close()


def plot_2d_segmentation_series(path: str,
                                file_name_suffix: str,
                                image: sitk.Image,
                                ground_truth : sitk.Image,
                                segmentation : sitk.Image,
                                alpha: float=0.5,
                                label: int=1,
                                file_extension: str='.png') -> None:
    """Plots an image with an overlaid mask, which indicates under-, correct-, and over-segmentation.

    Args:
        path (str): The output directory path.
        file_name_suffix (str): The output file name suffix.
        image (sitk.Image): The image.
        ground_truth (sitk.Image): The ground truth.
        segmentation (sitk.Image): The segmentation.
        alpha (float): The alpha blending value, between 0 (transparent) and 1 (opaque).
        label (int): The ground truth and segmentation label.
        file_extension (str): The output file extension (with or without dot).

    Examples:
        >>> img = sitk.ReadImage('your/path/image.mha')
        >>> ground_truth = sitk.ReadImage('your/path/ground_truth.mha')
        >>> segmentation = sitk.ReadImage('your/path/segmentation.mha')
        >>> plot_2d_segmentation_series('/your/path/', 'mysegmentation', img, ground_truth, segmentation)
    """

    if not image.GetSize() == ground_truth.GetSize() == segmentation.GetSize():
        raise ValueError('image, ground_truth, and segmentation must have equal size')
    if not image.GetDimension() == 3:
        raise ValueError('only 3-dimensional images supported')
    if not image.GetNumberOfComponentsPerPixel() == 1:
        raise ValueError('only scalar images supported')

    img_arr = sitk.GetArrayFromImage(image)
    gt_arr = sitk.GetArrayFromImage(ground_truth)
    seg_arr = sitk.GetArrayFromImage(segmentation)

    os.makedirs(path, exist_ok=True)
    file_extension = file_extension if file_extension.startswith('.') else '.' + file_extension

    for slice in range(img_arr.shape[0]):
        full_file_path = os.path.join(path, file_name_suffix + str(slice) + file_extension)
        plot_2d_segmentation(full_file_path,
                             img_arr[slice, ...],
                             gt_arr[slice, ...],
                             seg_arr[slice, ...],
                             alpha=alpha,
                             label=label)
