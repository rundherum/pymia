import collections

import torch.utils.data.dataloader as loader

import pymia.deeplearning.conversion as conv


class TorchCollate(conv.CollateBase):

    def __call__(self, batch) -> dict:
        """Merges the `entries` to Torch Tensors and the remaining entries in `batch` to a tuple.

        Args:
            batch (:obj:`list` of :obj:`dict`): A list of batch entries where len(batch) is the batch size.

        Returns:
            The merged batch, i.e. a dict of Torch Tensors and tuples.

        Examples:
            Use the class during the instantiation of a DataLoader:

            >>> import pymia.data.extraction as pymia_extr
            >>> collate = TensorFlowCollate(entries=('images', 'labels'))
            >>> loader = pymia_extr.DataLoader(dataset=None,collate_fn=collate)  # specify an appropriate dataset
        """
        if not isinstance(batch[0], collections.Mapping):
            raise TypeError('only mapping type allowed. Found {}'.format(type(batch[0])))

        new_batch = {}
        for key in batch[0]:
            if key in self.entries:
                new_batch[key] = loader.default_collate([d[key] for d in batch])
            else:
                new_batch[key] = [d[key] for d in batch]
        return new_batch