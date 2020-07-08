.. _examples:

Examples
========

The following examples illustrate the intended use of pymia:

.. toctree::
    :maxdepth: 1

    examples.data.creation
    examples.data.extraction_assembly
    examples.data.metadata_dataset
    examples.evaluation.basic
    examples.evaluation.logging
    examples.filtering.basic

The examples are available as Jupyter notebooks and Python scripts on `GitHub <https://github.com/rundherum/pymia/tree/master/examples/>`_) or directly rendered in the documentation by following the links above.
For all examples, 3 tesla MR images of the head of four healthy subjects from the Human Connectome Project (HCP)~\cite{VanEssen2013} are used.
Each subject has four 3-D images (in the MetaImage and Nifty format) and demographic information provided as a text file.
The images are a T1-weighted MR image, a T2-weighted MR image, a label image (ground truth), and a brain mask image.
The demographic information is artificially created age, gender, and grade point average (GPA).
The label images contain annotations of five brain structures (white matter, grey matter, hippocampus, amygdala, and thalamus), automatically segmented by FreeSurfer 5.3~\cite{Fischl2012,Fischl2002}.
Therefore, the examples mimic the problem of medical image segmentation of brain tissues.