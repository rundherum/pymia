"""This module holds classes related to batch feeding together with the :py:class`DataLoader`."""

class CollateBase:
    """Collate base class for merging a list of samples to a batch."""

    def __init__(self, entries=('images', 'labels')):
        """Initializes a new instance of the CollateBase class.

        Args:
            entries (:obj:`list` of :obj:`str`): The entries to merge for batch feeding to the GPU.
        """
        self.entries = entries

    def __call__(self, batch) -> dict:
        raise NotImplementedError('Derived class needs to implement __call__')

