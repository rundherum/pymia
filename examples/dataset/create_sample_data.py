import argparse
import os

import numpy as np
import SimpleITK as sitk


def create_and_save_image(path: str, shape: tuple, low: int, high: int):
    img_arr = np.random.randint(low, high + 1, shape).astype(np.int16)
    img = sitk.GetImageFromArray(img_arr)
    sitk.WriteImage(img, path)


def main(data_dir: str):
    N = 4  # number of subjects
    XY = 256  # X- = Y-dimension
    Z = 10  # Z-dimension (slices)

    np_shape = (Z, XY, XY)

    for n in range(N):
        np.random.seed(n)  # for reproducibility

        subject = 'Subject_{}'.format(n)
        subject_dir = os.path.join(data_dir, subject)

        os.makedirs(subject_dir, exist_ok=True)

        # save two MRI images, a ground truth, and a mask
        create_and_save_image(os.path.join(subject_dir, '{}_T1.mha'.format(subject)), np_shape, 0, 255)
        create_and_save_image(os.path.join(subject_dir, '{}_T2.mha'.format(subject)), np_shape, 0, 255)
        create_and_save_image(os.path.join(subject_dir, '{}_GT.mha'.format(subject)), np_shape, 0, 4)
        create_and_save_image(os.path.join(subject_dir, '{}_MASK.nii.gz'.format(subject)), np_shape, 0, 1)

        additional = 'Age: {}\nGPA: {}\nSex: {}'.format(np.random.randint(20, 90), np.random.rand(1)[0] * 4,
                                                        'M' if n % 2 == 0 else 'F')

        with open(os.path.join(subject_dir, 'demographic.txt'), 'w') as file:
            file.write(additional)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Sample data creation')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='out/test',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.data_dir)
