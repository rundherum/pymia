import SimpleITK as sitk


class ImageInformation:
    """
    Represents an image information.
    Also refer to itk::simple::Image::CopyInformation.
    """

    def __init__(self, image: sitk.Image):
        """
        Initializes a new instance of the ImageInformation class.
        :param image: The image.
        """
        self.size = image.GetSize()
        self.origin = image.GetOrigin()
        self.spacing = image.GetSpacing()
        self.direction = image.GetDirection()

    def is_two_dimensional(self) -> bool:
        return len(self.size) == 2

    def is_three_dimensional(self) -> bool:
        return len(self.size) == 3

    def is_three_dimensional_vector(self) -> bool:
        return len(self.size) == 4


class NumpySimpleITKImageBridge:
    """
    Represents a numpy to SimpleITK bridge.
    It provides static methods to convert between numpy array and SimpleITK image.
    """

    @staticmethod
    def array_to_image(array, information: ImageInformation) -> sitk.Image:
        """
        Converts a one-dimensional numpy array to a SimpleITK image.
        :param array: The image as numpy one-dimensional array, e.g. shape=(n,), where n = total number of voxels.
        :param information: The image information.
        :return: The SimpleITK image. 
        """

        if array.ndim != 1:
            raise ValueError("array needs to be one-dimensional")

        array = array.reshape((information.size[2], information.size[1], information.size[0]))

        image = sitk.GetImageFromArray(array)
        image.SetOrigin(information.origin)
        image.SetSpacing(information.spacing)
        image.SetDirection(information.direction)

        return image

    @staticmethod
    def array_to_vector_image(array, information: ImageInformation) -> sitk.Image:
        """
        Converts a two-dimensional numpy array to a SimpleITK vector image.
        :param array: The image as numpy two-dimensional array, e.g. shape=(4181760,2).
        :param information: The image information.
        :return: The SimpleITK image. 
        """

        if array.ndim != 2:
            raise ValueError("array needs to be two-dimensional")

        array = array.reshape((information.size[2], information.size[1], information.size[0], array.shape[1]))

        image = sitk.GetImageFromArray(array)
        image.SetOrigin(information.origin)
        image.SetSpacing(information.spacing)
        image.SetDirection(information.direction)

        return image


class SimpleITKNumpyImageBridge:
    """
    Represents a SimpleITK to numpy bridge.
    It provides static methods to convert between SimpleITK image and numpy array.
    """

    @staticmethod
    def convert(image: sitk.Image):
        """
        Converts 
        :param image: The image. 
        :return: (array, info): The image as numpy array and the `ImageInformation`.
        """

        return sitk.GetArrayFromImage(image), ImageInformation(image)
