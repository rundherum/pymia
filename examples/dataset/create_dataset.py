import argparse
import enum
import os
import typing

import SimpleITK as sitk

import miapy.data as miapy_data
import miapy.data.creation as miapy_crt
import miapy.data.loading as miapy_load
import miapy.data.transformation as miapy_tfm


class FileTypes(enum.Enum):
    T1 = 1  # The T1-weighted image
    T2 = 2  # The T2-weighted image
    GT = 3  # The ground truth image
    MASK = 4  # The foreground mask
    AGE = 5  # The age
    SEX = 6  # The sex


class Subject(miapy_data.SubjectFile):

    def __init__(self, subject: str, files: dict):
        super().__init__(subject, {})
        self.subject_path = files.pop(subject, '')
        self.sequence_files = {FileTypes.T1.name: files[FileTypes.T1], FileTypes.T2.name: files[FileTypes.T2]}
        self.gt_files = {FileTypes.GT.name: files[FileTypes.GT]}
        self.mask_files = {FileTypes.MASK.name: files[FileTypes.MASK]}


class DataSetFilePathGenerator(miapy_load.FilePathGenerator):
    """Represents a brain image file path generator.

    The generator is used to convert a human readable image identifier to an image file path,
    which allows to load the image.
    """

    def __init__(self):
        """Initializes a new instance of the DataSetFilePathGenerator class."""
        pass

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        """Gets the full file path for an image.
        Args:
            id_ (str): The image identification.
            root_dir (str): The image' root directory.
            file_key (object): A human readable identifier used to identify the image.
            file_extension (str): The image' file extension.
        Returns:
            str: The images' full file path.
        """
        add_file_extension = True

        if file_key == FileTypes.T1:
            file_name = '{}_T1'.format(id_)
        elif file_key == FileTypes.T2:
            file_name = '{}_T2'.format(id_)
        elif file_key == FileTypes.GT:
            file_name = '{}_GT'.format(id_)
        elif file_key == FileTypes.MASK:
            file_name = '{}_MASK.nii.gz'.format(id_)
            add_file_extension = False
        elif file_key == FileTypes.AGE or file_key == FileTypes.SEX:
            file_name = 'demographic.txt'
            add_file_extension = False
        else:
            raise ValueError('Unknown key')

        file_name = file_name + file_extension if add_file_extension else file_name
        return os.path.join(root_dir, file_name)


class DirectoryFilter(miapy_load.DirectoryFilter):
    """Represents a data directory filter."""

    def __init__(self):
        """Initializes a new instance of the DataDirectoryFilter class."""
        pass

    @staticmethod
    def filter_directories(dirs: typing.List[str]) -> typing.List[str]:
        """Filters a list of directories.
        Args:
            dirs (List[str]): A list of directories.
        Returns:
            List[str]: The filtered list of directories.
        """

        # currently, we do not filter the directories. but you could filter the directory list like this:
        # return [dir for dir in dirs if not dir.lower().__contains__('atlas')]
        return sorted(dirs)


class DataSetLoader(miapy_crt.Loader):

    def load_image(self, file_name: str, id_: str=None):
        return sitk.ReadImage(file_name, sitk.sitkFloat32)

    def load_label(self, file_name: str, id_: str=None):
        return sitk.ReadImage(file_name, sitk.sitkUInt8)

    def load_additional(self, file_name: str):  # todo we need enum here...
        return None  # todo: loading additional items

    def get_ndarray(self, image):
        return sitk.GetArrayFromImage(image)


def main(hdf_file: str, data_dir: str):
    keys = [FileTypes.T1, FileTypes.T2, FileTypes.GT, FileTypes.MASK, FileTypes.AGE, FileTypes.SEX]
    crawler = miapy_load.FileSystemDataCrawler(data_dir,
                                               keys,
                                               DataSetFilePathGenerator(),
                                               DirectoryFilter(),
                                               '.mha')

    subjects = [Subject(id_, file_dict) for id_, file_dict in crawler.data.items()]

    with miapy_crt.Hdf5Writer(hdf_file) as writer:
        callbacks = miapy_crt.ComposeCallback([miapy_crt.WriteDataCallback(writer),
                                               miapy_crt.WriteFilesCallback(writer),
                                               miapy_crt.WriteNamesCallback(writer),
                                               miapy_crt.WriteSubjectCallback(writer)])

        transform = miapy_tfm.IntensityRescale(0, 1)
        #transform = miapy_tfm.Compose([])

        traverser = miapy_crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, loader=DataSetLoader(), transform=transform)

if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='/home/fbalsiger/Documents/test/test.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/fbalsiger/Documents/test',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)
