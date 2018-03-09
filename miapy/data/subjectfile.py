class SubjectFile:

    def __init__(self, subject: str, images: dict, label_images: dict=None, supplementaries: dict=None) -> None:
        self.subject = subject
        self.images = images
        self.label_images = label_images
        self.supplementaries = supplementaries
        self._check_validity()

    def _check_validity(self):
        all_keys = list(self.images.keys())
        if self.label_images is not None:
            all_keys.extend(self.label_images.keys())
        if self.supplementaries is not None:
            all_keys.extend(self.supplementaries.keys())

        if len(all_keys) > len(set(all_keys)):
            raise ValueError('Identifiers must be unique')

    def get_all_files(self):
        labels = {} if self.label_images is None else self.label_images
        supplementaries = {} if self.supplementaries is None else self.supplementaries
        return {**self.images, **labels, **supplementaries}
