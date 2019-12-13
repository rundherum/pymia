import numpy as np
import torch.utils.data.dataloader as loader

import pymia.deeplearning.conversion as conv


class TensorFlowCollate(conv.CollateBase):

    def __call__(self, batch) -> dict:
        """Merges the `entries` to numpy arrays and the remaining entries in `batch` to a tuple.

        Args:
            batch (:obj:`list` of :obj:`dict`): A list of batch entries where len(batch) is the batch size.

        Returns:
            The merged batch, i.e. a dict of numpy arrays and tuples.

        Examples:
            Use the class during the instantiation of a DataLoader:

            >>> import pymia.data.extraction as pymia_extr
            >>> collate = TensorFlowCollate(entries=('images', 'labels'))
            >>> loader = pymia_extr.DataLoader(dataset=None,collate_fn=collate)  # specify an appropriate dataset
        """
        new_batch = {}
        for key in batch[0]:
            if key in self.entries:
                new_batch[key] = np.array([d[key] for d in batch])
            else:
                new_batch[key] = tuple([d[key] for d in batch])
        return new_batch