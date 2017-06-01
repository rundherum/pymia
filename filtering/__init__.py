"""
===================================================
Image filter and manipulation (:mod:`filtering`)
===================================================

.. currentmodule:: filtering
This package contains various image filters and image
manipulation functions.

Smoothing :mod:`medpy.filter.smoothing`
=======================================
Image smoothing / noise reduction in grayscale images.
.. module:: medpy.filter.smoothing
.. autosummary::
    :toctree: generated/

    anisotropic_diffusion
    gauss_xminus1d

Binary :mod:`medpy.filter.binary`
=================================
Binary image manipulation.
.. module:: medpy.filter.binary
.. autosummary::
    :toctree: generated/

    size_threshold
    largest_connected_component
    bounding_box

Image :mod:`medpy.filter.image`
=================================
Grayscale image manipulation.
.. module:: medpy.filter.image
.. autosummary::
    :toctree: generated/

    sls
    ssd
    average_filter
    sum_filter
    local_minima
    otsu
    resample

Intensity range standardization :mod:`medpy.filter.IntensityRangeStandardization`
=================================================================================
A learning method to align the intensity ranges of images.
.. module:: medpy.filter.IntensityRangeStandardization
.. autosummary::
    :toctree: generated/

    IntensityRangeStandardization
"""