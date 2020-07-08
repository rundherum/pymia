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

The examples are available as Jupyter notebooks and Python scripts on `GitHub <https://github.com/rundherum/pymia/tree/master/examples/>`_ or directly rendered in the documentation by following the links above.
For all examples, 3 tesla MR images of the head of four healthy subjects from the Human Connectome Project (HCP) [VanEssen2013]_ are used.
Each subject has four 3-D images (in the MetaImage and Nifty format) and demographic information provided as a text file.
The images are a T1-weighted MR image, a T2-weighted MR image, a label image (ground truth), and a brain mask image.
The demographic information is artificially created age, gender, and grade point average (GPA).
The label images contain annotations of five brain structures (white matter, grey matter, hippocampus, amygdala, and thalamus), automatically segmented by FreeSurfer 5.3 [Fischl2012]_ [Fischl2002]_.
Therefore, the examples mimic the problem of medical image segmentation of brain tissues.

.. [VanEssen2013] Van Essen, D. C., Smith, S. M., Barch, D. M., Behrens, T. E. J., Yacoub, E., Ugurbil, K., & WU-Minn HCP Consortium. (2013). The WU-Minn Human Connectome Project: An overview. NeuroImage, 80, 62–79. https://doi.org/10.1016/j.neuroimage.2013.05.041

.. [Fischl2012] Fischl, B. (2012). FreeSurfer. NeuroImage, 62(2), 774–781. https://doi.org/10.1016/j.neuroimage.2012.01.021

.. [Fischl2002] Fischl, B., Salat, D. H., Busa, E., Albert, M., Dieterich, M., Haselgrove, C., … Dale, A. M. (2002). Whole brain segmentation: Automated labeling of neuroanatomical structures in the human brain. Neuron, 33(3), 341–355. https://doi.org/10.1016/S0896-6273(02)00569-X