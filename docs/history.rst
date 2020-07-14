.. _history:

Change history
==============

0.3.0 (2020-07-14)
------------------

 * :mod:`pymia.data` package now supports PyTorch and TensorFlow. A few classes have been renamed and refactored.
 * :mod:`pymia.evaluation` package with new evaluator and writer classes. Metrics are now categorized into :mod:`pymia.evaluation.metric.categorical` and :mod:`pymia.evaluation.metric.continuous` modules
 * New metrics :class:`.PeakSignalToNoiseRatio` and :class:`.StructuralSimilarityIndexMeasure`
 * Removed ``config``, ``deeplearning``, and ``plotting`` packages
 * Improved readability of code
 * Revised examples
 * Revised documentation

Migration guide
^^^^^^^^^^^^^^^

Heavy changes have been made to move pymia towards a lightweight data handling and evaluation package for
medical image analysis with deep learning. Therefore, this release is, unfortunately, not backward compatible.
To facilitate transition to this and coming versions, we thoroughly revised the documentation and the :ref:`examples <examples>`.

0.2.4 (2020-05-22)
------------------

 * Bug fixes in the :mod:`pymia.evaluation` package

0.2.3 (2019-12-13)
------------------

 * Refactored: :mod:`pymia.data.transformation`
 * Bug fixes and code maintenance


0.2.2 (2019-11-11)
------------------

 * Removed the ``tensorflow``, ``tensorboardX``, and ``torch`` dependencies during installation
 * Bug fixes and code maintenance

0.2.1 (2019-09-04)
------------------

 * New statistics plotting module :mod:`pymia.plotting.statistics` (subject to heavy changes and possibly removal!)
 * Bug fixes and code maintenance
 * Several improvements to the documentation

0.2.0 (2019-04-12)
------------------

 * New :mod:`pymia.deeplearning` package
 * New extractor :class:`.PadDataExtractor`, which replaces the ``PadPatchDataExtractor`` (see migration guide below)
 * New metrics :class:`.NormalizedRootMeanSquaredError`, :class:`.SurfaceDiceOverlap`, and :class:`.SurfaceOverlap`
 * Faster and more generic implementation of :class:`.HausdorffDistance`
 * New data augmentation module :mod:`pymia.data.augmentation`
 * New filter :class:`.BinaryThreshold`
 * Replaced the transformation in :class:`.SubjectAssembler` by a more flexible function (see migration guide below)
 * Minor bug fixes and maintenance
 * Several improvements to the documentation

We kindly appreciate the help of our contributors:

 - Jan Riedo
 - Yannick Soom

Migration guide
^^^^^^^^^^^^^^^

The extractor ``PadPatchDataExtractor`` has been replaced by the :class:`.PadDataExtractor` to facilitate the
extraction flexibility. The :class:`.PadDataExtractor` works now with any kind of the three data extractors
(:class:`.DataExtractor`, :class:`.RandomDataExtractor`, and :class:`.SelectiveDataExtractor`),
which are passed as argument. Further, it is now possible to pass a function for the padding as argument to replace the
default zero padding. Suppose you used the ``PadPatchDataExtractor`` like this:

.. code-block:: python

  import pymia.data.extraction as pymia_extr
  pymia_extr.PadPatchDataExtractor(padding=(10, 10, 10), categories=('images',))

To have the same behaviour, replace it by:

.. code-block:: python

  import pymia.data.extraction as pymia_extr
  pymia_extr.PadDataExtractor(padding=(10, 10, 10),
                              extractor=pymia_extr.DataExtractor(categories=('images',)))

The transformation in :meth:`.SubjectAssembler.add_batch` has been removed and replaced by the ``on_sample_fn``
parameter in the constructor. Replacing the transformation by this function should be straight forward by rewriting your
transformation as function:

.. code-block:: python

  def on_sample_fn(params: dict):
    key = '__prediction'
    batch = params['batch']
    idx = params['batch_idx']

    data = params[key]
    index_expr = batch['index_expr'][idx]

    # manipulate data and index_expr according to your needs

    return data, index_expr

0.1.1 (2018-08-04)
------------------

 * Improves the documentation
 * Mocks the torch dependency to build the docs

0.1.0 (2018-08-03)
------------------

 * Initial release on PyPI
