How to use the `miapy` for deep learning projects
=================================================

This example explains step by step how to use `miapy` for deep learning projects. `miapy` itself is not a deep learning library
but provides packages to facilitate your projects. We are working with medical images, so first let's create some (realistic) sample data for a medical image segmentation task.

.. literalinclude:: ../examples/dataset/create_sample_data.py

Executing this script will generate sample data for four subjects. Each subject will have four three-dimensional images and demographic information.
The images are a T1-weighted, a T2-weighted, a ground truth (GT), and a mask image. The demographic information are age, sex, and grade point average (GPA).

Next, we will create a data set from the sample data, which allows us to have fast, easy, and flexible access to our data.

.. literalinclude:: ../examples/dataset/create_dataset.py

Executing this script will generate a H5 file.

.. note::

    You can use `verify_dataset.py` located under `./examples/dataset` to loop over the data set and verify it.

Next, we will create a configuration file and corresponding code to configure our project.

.. literalinclude:: ../examples/dataset/config.json

The above lines represent a JSON configuration file. As you can see, we have four configuration entries:

* database_file: The H5 database file generated above.
* batch_size_training: The batch size during the training phase.
* batch_size_testing: The batch size during the tesing phase.
* epochs: The number of epochs to train.

Next, we will define a Python module to parse this JSON configuration file and use it in code.

.. literalinclude:: ../examples/dataset/config.py

The above module allows us to parse the JSON configuration file and use it as Configuration class in code.

Next, we will define a main Python script for training and tesing our deep learning algorithm, which uses the H5 database file, the configuration file, and the configuration module.

.. literalinclude:: ../examples/dataset/main.py

Wohoo, a lot is happening here. Let's break it down.

TODO