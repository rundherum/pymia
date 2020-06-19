.. _example-data1-old:

How to use the pymia for deep learning projects - Part 1: Handle your data
==========================================================================

This example explains step by step how to use `pymia` for data handling in deep learning projects. `pymia` itself is not a deep learning library
but provides packages to facilitate your projects. We are working with medical images, so first let's create some dummy data for a medical image segmentation task.

.. literalinclude:: ../examples/data/create_dummy_data.py

Executing this script will generate dummy data of four subjects. Each subject will have four 3-D images (MetaImage format, .mha) and demographic information.
The images are a T1-weighted, a T2-weighted, a ground truth (GT), and a mask image. The demographic information are age, sex, and grade point average (GPA).

Next, we will create a dataset from the dummy data, which allows us to have easy, flexible, and fast access to our data.

.. literalinclude:: ../examples/data/create_dataset.py

Executing this script will generate a HDF5 file (dummy.h5), containing all the dummy data for simple access.

.. note::

    You can use `verify_dataset.py` located under `./examples/data` to loop over the dataset and verify it.

Next, we will define a main script for training and validation of our deep learning model, which uses the HDF5 dataset file as data source.

.. literalinclude:: ../examples/data/main.py

A lot is happening here, so let's break it down. First of all, we want to extract that data slice-wise. Therefore, the indexing strategy to extract the image slices from the dataset is

`indexing_strategy = pymia_extr.SliceIndexing()`