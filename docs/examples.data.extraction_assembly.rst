.. _example-data2:

Data extraction and assembly
============================

This example shows how to use the :mod:`pymia.data` package to extract chunks of data from the dataset and to assemble the chunks
to feed a deep neural network. It also shows how the predicted chunks are assembled back to full-images predictions.

The extraction-assemble principle is essential for large three-dimensional images that do not fit entirely in the GPU memory
and thus require some kind of patch-based approach.

For simplicity reasons we use slice-wise extraction in this example, meaning that the two-dimensional slices are extracted
from the three-dimensional image. Further, the example uses the PyTorch PyTorch as a deep learning (DL) framework.

At the end of this page you find examples for the following additional use cases:

* Using pymia with TensorFlow
* Extracting 3-D patches
* Extracting from a metadata dataset

.. note::
    To be able to run this example you need the example data. If you haven't already, you can obtain the example data by
    executing `./examples/example-data/pull_example_data.py`.


Code walkthrough
----------------

**[0]** Import the required modules. ::

    import pymia.data.assembler as assm
    import pymia.data.transformation as tfm
    import pymia.data.definition as defs
    import pymia.data.extraction as extr
    import pymia.data.backends.pytorch as pymia_torch

**[1]**  First, we create the the access to the .h5 dataset by defining: (i) the indexing strategy (`indexing_strategy`)
that defines the chunks of data to be retrieved, (ii) the information to be extracted (`extractor`), and (iii)
the transformation (`transform`) to be applied after extraction.

The permutation transform is required since the channels (here _T1_, _T2_) are stored in the last dimension in the .h5 dataset
but PyTorch requires channel-first format. ::

    hdf_file = '../example-data/example-dataset.h5'

    # Data extractor for extracting the "images" entries
    extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES,))
    # Permutation transform to go from HWC to CHW.
    transform = tfm.Permute(permutation=(2, 0, 1), entries=(defs.KEY_IMAGES,))
    # Indexing defining a slice-wise extraction of the data
    indexing_strategy = extr.SliceIndexing()

    dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, transform)


**[2]**  Next, we define an assembler that will puts the data/image chunks back together after prediction of the input chunks. This is
required to perform a evaluation on entire subjects, and any further processing such as saving the predictions.

Also, we define extractors that we will use to extract information required after prediction. This information not need
to be chunked (/indexed/sliced) and not need to interact with the DL framework. Thus, it can be extracted
directly form the dataset. ::

    assembler = assm.SubjectAssembler(dataset)

    direct_extractor = extr.ComposeExtractor([
        extr.ImagePropertiesExtractor(),  # Extraction of image properties (origin, spacing, etc.) for storage
        extr.DataExtractor(categories=(defs.KEY_LABELS,))  # Extraction of "labels" entries for evaluation
    ])


**[3]**  The batch generation and and the neural network architecture are framework dependent.
Basically, all we have to do is to wrap our dataset as PyTorch dataset, to build a PyTorch data loader, and to create/load a
network. ::

    import torch
    import torch.nn as nn
    import torch.utils.data as torch_data

    # Wrap the pymia datasource
    pytorch_dataset = pymia_torch.PytorchDatasetAdapter(dataset)
    loader = torch_data.dataloader.DataLoader(pytorch_dataset, batch_size=2, shuffle=False)

    # Dummy network representing a placeholder for a trained network
    dummy_network = nn.Sequential(
        nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
        nn.Sigmoid()
    ).eval()
    torch.set_grad_enabled(False)  # no gradients needed for testing

    nb_batches = len(loader)


**[4]**  We are now ready to loop over batches of data chunks. After the usual prediction of the network, the predicted data is
provided to the assembler, which takes care of putting chunks back together. Once some subjects are assembled
(`subjects_ready`) we extract the data required for evaluation and storing. ::

    for i, batch in enumerate(loader):

        # Get data from batch and predict
        x, sample_indices = batch[defs.KEY_IMAGES], batch[defs.KEY_SAMPLE_INDEX]
        prediction = dummy_network(x)

        # translate the prediction to numpy and back to (B)HWC (channel last)
        numpy_prediction = prediction.numpy().transpose((0, 2, 3, 1))

        # add the batch prediction to the assembler
        is_last = i == nb_batches - 1
        assembler.add_batch(numpy_prediction, sample_indices.numpy(), is_last)

        # Process the subjects/images that are fully assembled
        for subject_index in assembler.subjects_ready:
            subject_prediction = assembler.get_assembled_subject(subject_index)

            # Extract the target and image properties via direct extract
            direct_sample = dataset.direct_extract(direct_extractor, subject_index)
            target, image_properties = direct_sample[defs.KEY_LABELS], direct_sample[defs.KEY_PROPERTIES]

            # # Do whatever you desire...
            # do_eval()
            # do_save()

Other use cases
---------------


Using pymia with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only the :class:`.PymiaDatasource` wrapping has to be changed to use the pymia data handling together with TensorFlow instead
of PyTorch. This change, however, implies other framework related changes.

**[0]** Add Tensorflow specific imports. ::

    import tensorflow as tf
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    import pymia.data.backends.tensorflow as pymia_tf


**[1]** Wrap the :class:`.PymiaDatasource` (`dataset`) and use Tensorflow specific data handling. ::

    gen_fn = pymia_tf.get_tf_generator(dataset)
    tf_dataset = tf.data.Dataset.from_generator(generator=gen_fn,
                                                output_types={defs.KEY_IMAGES: tf.float32,
                                                              defs.KEY_SAMPLE_INDEX: tf.int64})
    loader = tf_dataset.batch(2)

    dummy_network = keras.Sequential([
        layers.Conv2D(8, kernel_size=3, padding='same'),
        layers.Conv2D(2, kernel_size=3, padding='same', activation='sigmoid')]
    )
    nb_batches = len(dataset) // 2

**[2]** As opposed to PyTorch, Tensorflow uses the channel-last (BWHC) configuration.
Thus, the permutations are no longer required ::

    # The lines following lines of the initial code ...
    transform = tfm.Permute(permutation=(2, 0, 1), entries=(defs.KEY_IMAGES,))
    numpy_prediction = prediction.numpy().transpose((0, 2, 3, 1))
    # ... become
    transform = None
    numpy_prediction = prediction.numpy()


Extracting 3-D patches
~~~~~~~~~~~~~~~~~~~~~~

To extract 3-D patches instead of slices requires only a few changes.

**[0]** Modifications on the indexing are typically due to a network change. Here, we still use a dummy network, but this time
it consists of 3-D valid convolutions (instead of 2-D same convolutions). ::

    dummy_network = nn.Sequential(
        nn.Conv3d(in_channels=2, out_channels=8, kernel_size=3, padding=0),
        nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, padding=0),
        nn.Sigmoid()
    )

**[1]** By knowing the architecture of the new network, we can modify the pymia related extraction. Note that the network
input shape is by 4 voxels larger then the output shape (valid convolutions). A input patch size of 36x36x36
extracted and the output patch size will be 32x32x32. ::

    # Adapted permutation due to the additional dimension
    transform = tfm.Permute(permutation=(3, 0, 1, 2), entries=(defs.KEY_IMAGES,))

    # Use a pad extractor to compensate input-output shape difference of the network. Actual image information is padded.
    extractor = extr.PadDataExtractor((2, 2, 2), extr.DataExtractor(categories=(defs.KEY_IMAGES,)))


**[2]** The modifications from 2-D to 3-D also affects the permutations. ::

    transform = tfm.Permute(permutation=(3, 0, 1, 2), entries=(defs.KEY_IMAGES,))
    numpy_prediction = prediction.numpy().transpose((0, 2, 3, 4, 1))


Extracting from a metadata dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A metadata dataset only contains metadata but not image (or other) data. Metadata datasets might be used when the amount of
data is large. They avoid storing a copy of the data in the dataset and access the raw data directly via the file links.

Extracting data from a metadata dataset is very simple and only requires to employ the corresponding :class:`.Extractor`. ::

    # The following line of the initial code ...
    extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES,))
    # ... becomes
    extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,))

