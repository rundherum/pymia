"""Enables the plotting of image histograms."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import typing as t
import SimpleITK as sitk


def plot_histogram(path: str, image: t.Union[sitk.Image, np.ndarray], no_bins: int=255, use_percentages: bool=False,
                   title: str='', xlabel: str='', ylabel: str='', slice_no: int=-1) -> None:
    """Plots a histogram of an image.

    Plots either the histogram of a slice of the image or of the whole image.

    Args:
        path (str): The file path.
        image (SimpleITK.Image or numpy array): The image.
        no_bins (int): The number of histogram bins.
        use_percentages (bool): Use percentages on the y axis if True; otherwise, use the number of values in the bins.
        title (str): The histogram's title.
        xlabel (str): The histogram's x-axis label.
        ylabel (str): The histogram's y-axis label.
        slice_no (int): The slice number or -1 to take the whole image (only supported for SimpleITK images).
    """

    if isinstance(image, sitk.Image):
        if slice_no > -1:
            data = sitk.GetArrayFromImage(image[:, :, slice_no])
        else:
            data = sitk.GetArrayFromImage(image)
    else:
        data = image

    data = data.flatten()

    plt.hist(data, bins=no_bins, density=use_percentages)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)

    if use_percentages:
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.savefig(path)
    plt.close()


def plot_histogram_overlay(path: str, image1: sitk.Image, image2: sitk.Image,
                           legend1: str = '', legend2: str = '',
                           no_bins: int = 255, slice_no: int = -1,
                           title: str = '', xlabel: str = '', ylabel: str = '',
                           xaxis_log: bool = False, yaxis_log: bool = False) -> None:
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
