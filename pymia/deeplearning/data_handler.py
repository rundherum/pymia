import abc


class DataHandler(abc.ABC):
    """Represents an abstract data handler for the pymia data package."""

    def __init__(self):
        self.dataset = None  # pymia_extr.ParameterizableDataset

        self.extractor_train = None  # pymia_extr.Extractor
        self.extractor_valid = None  # pymia_extr.Extractor
        self.extractor_test = None  # pymia_extr.Extractor

        self.extraction_transform_train = None  # pymia_tfm.Transform
        self.extraction_transform_valid = None  # pymia_tfm.Transform
        self.extraction_transform_test = None  # pymia_tfm.Transform

        self.loader_train = None  # torch.utils.data.DataLoader
        self.loader_valid = None  # torch.utils.data.DataLoader
        self.loader_test = None  # torch.utils.data.DataLoader

        self.no_subjects_train = 0
        self.no_subjects_valid = 0
        self.no_subjects_test = 0
