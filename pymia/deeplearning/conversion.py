import collections

import torch.utils.data.dataloader as loader


def collate_batch(batch: list) -> dict:
    """Collate function for PyTorch DataLoader class.

    Converts a list of dicts to a dict of lists.
    """
    return dict(zip(batch[0], zip(*[d.values() for d in batch])))


class TorchCollate:

    def __init__(self, entries=('images', 'labels')):
        self.entries = entries

    def __call__(self, batch):
        if not isinstance(batch[0], collections.Mapping):
            raise TypeError('only mapping type allowed. Found {}'.format(type(batch[0])))

        new_batch = {}
        for key in batch[0]:
            if key in self.entries:
                new_batch[key] = loader.default_collate([d[key] for d in batch])
            else:
                new_batch[key] = [d[key] for d in batch]
        return new_batch
