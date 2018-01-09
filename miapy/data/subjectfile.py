import abc
import typing as t


class FileCollection(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_sequences(self) -> t.Dict[str, str]:
        pass

    @abc.abstractmethod
    def get_gts(self) -> t.Dict[str, str]:
        pass

    @abc.abstractmethod
    def get_subject(self) -> str:
        pass


class SubjectFile(FileCollection):

    def __init__(self, subject: str, sequence_files: dict, gt_files: dict=None) -> None:
        self.subject = subject
        self.sequence_files = sequence_files
        self.gt_files = gt_files

    def get_sequences(self) -> t.Dict[str, str]:
        return self.sequence_files

    def get_gts(self) -> t.Dict[str, str]:
        return self.gt_files

    def get_subject(self) -> str:
        return self.subject
