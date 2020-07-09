import torch.utils.data.sampler as smplr


class SubsetSequentialSampler(smplr.Sampler):

    def __init__(self, indices):
        """Samples elements sequential from a given list of indices, without replacement.

        The class adopts the `troch.utils.data.Sampler
        <https://pytorch.org/docs/1.3.0/data.html#torch.utils.data.Sampler>`_ interface.

        Args:
            indices list: list of indices that define the subset to be used for the sampling.
        """
        super().__init__(None)
        self.indices = indices

    def __iter__(self):
        return (idx for idx in self.indices)

    def __len__(self):
        return len(self.indices)
