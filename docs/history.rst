.. _history:

Change history
==============

0.2.4 (2020-05-22)
------------------

 * Bug fixes in the :py:mod:`evaluation` package

0.2.3 (2019-12-13)
------------------

 * Refactored: :py:mod:`transformation`
 * Bug fixes and code maintenance


0.2.2 (2019-11-11)
------------------

 * Removed the ``tensorflow``, ``tensorboardX``, and ``torch`` dependencies during installation
 * Bug fixes and code maintenance

0.2.1 (2019-09-04)
------------------

 * New statistics plotting module :py:mod:`statistics` (subject to heavy changes and possibly removal!)
 * Bug fixes and code maintenance
 * Several improvements to the documentation

0.2.0 (2019-04-12)
------------------

 * New :py:mod:`deeplearning` package
 * New extractor :py:class:`PadDataExtractor`, which replaces the ``PadPatchDataExtractor`` (see migration guide below)
 * New metrics :py:class:`NormalizedRootMeanSquaredError`, :py:class:`SurfaceDiceOverlap`, and :py:class:`SurfaceOverlap`
 * Faster and more generic implementation of :py:class:`HausdorffDistance`
 * New data augmentation module :py:mod:`augmentation`
 * New filter :py:class:`BinaryThreshold`
 * Replaced the transformation in :py:mod:`SubjectAssembler` by a more flexible function (see migration guide below)
 * Minor bug fixes and maintenance
 * Several improvements to the documentation

We kindly appreciate the help of our contributors:

 - Jan Riedo
 - Yannick Soom

Migration guide
^^^^^^^^^^^^^^^

The extractor ``PadPatchDataExtractor`` has been replaced by the :py:class:`PadDataExtractor` to facilitate the
extraction flexibility. The :py:class:`PadDataExtractor` works now with any kind of the three data extractors
(:py:class:`DataExtractor`, :py:class:`RandomDataExtractor`, and :py:class:`SelectiveDataExtractor`),
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

The transformation in :py:mod:`SubjectAssembler`'s ``add_batch`` has been removed and replaced by the ``on_sample_fn``
parameter in the constructor. Replacing the transformation by this function should be straight forward by rewriting your
transformation as function (see also the default sample function :py:function:`default_sample_fn`):

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
