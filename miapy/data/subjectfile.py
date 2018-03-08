class SubjectFile:

    def __init__(self, subject: str, images: dict, label_images: dict=None, supplementaries: dict=None) -> None:
        self.subject = subject
        self.images = images
        self.label_images = label_images
        self.supplementaries = supplementaries
