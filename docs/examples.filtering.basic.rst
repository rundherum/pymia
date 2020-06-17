.. _example-filtering1:
Filter pipelines
================

This example shows how to use the filtering package to set up image filter and manipulation pipelines.

.. note::
   To be able to run this example:

    * Get the example data by executing `./examples/example-data/pull_example_data.py`.
    * Install matplotlib (`pip install matplotlib`)

The pipeline will apply a gradient anisotropic diffusion filter followed by a histogram matching to the T1-weighted MR images.
We will use the T2-weighted MR images as a reference for the histogram matching.

.. literalinclude:: ../examples/filtering/basic.py
