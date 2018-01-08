import abc
import typing as t


# todo: since this is the same as in uncertainty/data/transformation, find way to only define one

# follows the principle of torchvision transform
class Transform(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, sample: dict) -> dict:
        pass


class Compose(Transform):

    def __init__(self, transforms: t.Iterable[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample
