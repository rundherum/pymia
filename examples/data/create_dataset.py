import argparse
import enum
import glob
import os
import typing

import SimpleITK as sitk
import numpy as np

import pymia.data as data
import pymia.data.conversion as conv
import pymia.data.definition as defs
import pymia.data.creation as crt
import pymia.data.transformation as tfm
import pymia.data.creation.fileloader as file_load


class FileTypes(enum.Enum):
    T1 = 1  # The T1-weighted image
    T2 = 2  # The T2-weighted image
    GT = 3  # The ground truth image
    MASK = 4  # The foreground mask
    AGE = 5  # The age
    GPA = 6  # The GPA
    SEX = 7  # The sex


class LoadData(file_load.Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        if id_ == FileTypes.AGE.name:
            with open(file_name, 'r') as f:
                value = np.asarray([int(f.readline().split(':')[1].strip())])
                return value, None
        if id_ == FileTypes.GPA.name:
            with open(file_name, 'r') as f:
                value = np.asarray([float(f.readlines()[1].split(':')[1].strip())])
                return value, None
        if id_ == FileTypes.SEX.name:
            with open(file_name, 'r') as f:
                value = np.array(f.readlines()[2].split(':')[1].strip())
                return value, None

        if category == defs.KEY_IMAGES:
            img = sitk.ReadImage(file_name, sitk.sitkFloat32)
        else:
            # this is the ground truth (defs.KEY_LABELS), which will be loaded as unsigned integer
            img = sitk.ReadImage(file_name, sitk.sitkUInt8)

        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)


class Subject(data.SubjectFile):

    def __init__(self, subject: str, files: dict):
        super().__init__(subject,
                         images={FileTypes.T1.name: files[FileTypes.T1], FileTypes.T2.name: files[FileTypes.T2]},
                         labels={FileTypes.GT.name: files[FileTypes.GT]},
                         mask={FileTypes.MASK.name: files[FileTypes.MASK]},
                         numerical={FileTypes.AGE.name: files[FileTypes.AGE], FileTypes.GPA.name: files[FileTypes.GPA]},
                         sex={FileTypes.SEX.name: files[FileTypes.SEX]})
        self.subject_path = files.get(subject, '')


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
    elif file_key == FileTypes.AGE or file_key == FileTypes.GPA or file_key == FileTypes.SEX:
        file_name = 'demographic.txt'
        add_file_extension = False
    else:
        raise ValueError('Unknown key')

    file_name = file_name + file_extension if add_file_extension else file_name
    return os.path.join(root_dir, file_name)


def main(hdf_file: str, data_dir: str):
    # get subjects
    subject_dirs = [subject_dir for subject_dir in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(subject_dir)]
    sorted(subject_dirs)

    keys = [FileTypes.T1, FileTypes.T2, FileTypes.GT, FileTypes.MASK, FileTypes.AGE, FileTypes.GPA, FileTypes.SEX]
    subjects = []
    # for each subject on file system, initialize a Subject object
    for subject_dir in subject_dirs:
        id_ = os.path.basename(subject_dir)

        file_dict = {id_: subject_dir}  # init dict with id_ pointing to path
        for key in keys:
            file_path = get_full_file_path(id_, subject_dir, key, '.mha')
            file_dict[key] = file_path

        subjects.append(Subject(id_, file_dict))

    if os.path.exists(hdf_file):
        os.remove(hdf_file)

    with crt.get_writer(hdf_file) as writer:
        callbacks = crt.get_default_callbacks(writer)

        # add a transform to normalize the images
        transform = tfm.IntensityNormalization(loop_axis=3, entries=(defs.KEY_IMAGES, ))

        traverser = crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(), transform=transform)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../dummy-data/dummy.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../dummy-data',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)
