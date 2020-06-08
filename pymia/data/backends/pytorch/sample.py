import torch.utils.data.sampler as smplr


class SubsetSequentialSampler(smplr.Sampler):
    """Samples elements sequential from a given list of indices, without replacement."""

    def __init__(self, indices):
        super().__init__(None)
        self.indices = indices

    def __iter__(self):
        return (idx for idx in self.indices)

    def __len__(self):
        return len(self.indices)

