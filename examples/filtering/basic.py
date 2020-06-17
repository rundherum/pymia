import argparse
import glob
import os

import matplotlib.pyplot as plt
import pymia.filtering.filter as flt
import pymia.filtering.preprocessing as prep
import SimpleITK as sitk


def main(data_dir: str):
    # initialize the filtering pipeline, that is a gradient anisotropic filter followed by a histogram matching
    filters = [
        prep.GradientAnisotropicDiffusion(),
        prep.HistogramMatcher()
    ]
    pipeline = flt.FilterPipeline(filters)
    histogram_matching_filter_idx = 1

    # get list of subjects
    subject_dirs = [subject for subject in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(subject)]

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        print(f'Filtering {subject_id}...')

        # load the T1- and T2-weighted MR images
        t1_image = sitk.ReadImage(os.path.join(subject_dir, f'{subject_id}_T1.mha'))
        t2_image = sitk.ReadImage(os.path.join(subject_dir, f'{subject_id}_T2.mha'))

        # set the T2-weighted MR image as reference for the histogram matching
        pipeline.set_param(prep.HistogramMatcherParams(t2_image), histogram_matching_filter_idx)

        # execute filtering pipeline to the T1w image
        filtered_t1_image = pipeline.execute(t1_image)

        # plot filtering result
        slice_no_for_plot = t1_image.GetSize()[2] // 2
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(sitk.GetArrayFromImage(t1_image[:, :, slice_no_for_plot]), cmap='gray')
        axs[0].set_title('Original image')
        axs[1].imshow(sitk.GetArrayFromImage(filtered_t1_image[:, :, slice_no_for_plot]), cmap='gray')
        axs[1].set_title('Filtered image')
        fig.suptitle(f'{subject_id}', fontsize=16)
        plt.show()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Image filtering')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../example-data',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.data_dir)
