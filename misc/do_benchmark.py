import os
import time
import random

import numpy as np
import pandas as pd

import pymia.data.definition as defs
import pymia.data.extraction as extr


methods = ['dataset', 'file-numpy', 'file-standard', 'file-uncompressed']
tasks = ['slice', 'patch', 'full']
root_dir = 'benchmark_data/'


def main():

    nb_samples = 100

    entries_per_subject = 2
    nb_subjects_list = [5, 10, 25, 50, 100, 200]

    results = {}

    for nb_subjects in nb_subjects_list:
        in_dir = os.path.join(root_dir, f'datasets_subjects-{nb_subjects}_each-{entries_per_subject}')
        for method in methods:
            hdf_file = os.path.join(in_dir, f'benchmark_{method}.h5')
            assert os.path.isfile(hdf_file)

            for task in tasks:
                print(f'subjects: {nb_subjects}\tmethod: {method}\ttask{task}')
                indexing, extractor = get_indexing_and_extractor(task, method)
                dataset = extr.PymiaDatasource(hdf_file, indexing, extractor, transform=None)

                # with replacement
                sample_indices = random.Random(42).choices(list(range(len(dataset))), k=nb_samples)

                for i, sample_idx in enumerate(sample_indices):
                    start = time.time()
                    x = dataset[sample_idx][defs.KEY_IMAGES]
                    duration = time.time() - start

                    r = {'method': method, 'task': task, 'nb_subjects': nb_subjects, 'sample_idx': i, 'duration': duration}
                    for k, v in r.items():
                        results.setdefault(k, []).append(v)

    out_file = f'benchmark_result_each-{entries_per_subject}_new.csv'
    df = pd.DataFrame(data=results)
    df.to_csv(out_file, index=False)


def np_read(file_path, cat):
    return np.load(file_path)


def get_indexing_and_extractor(task, method):
    if task == 'slice':
        indexing = extr.SliceIndexing()
    elif task == 'patch':
        indexing = extr.PatchWiseIndexing((64, 64, 64))
    elif task == 'full':
        indexing = extr.EmptyIndexing()
    else:
        raise ValueError(f'unknown task "{task}"')

    if method == 'dataset':
        extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES,))
    elif method == 'file-numpy':
        extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,), load_fn=np_read)
    elif method == 'file-standard' or method == 'file-uncompressed':
        extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,))
    else:
        raise ValueError(f'unknown method "{method}"')

    if task == 'patch':
        padding = (10, 10, 10)
        extractor = extr.PadDataExtractor(padding, extractor)

    return indexing, extractor


if __name__ == '__main__':
    main()
