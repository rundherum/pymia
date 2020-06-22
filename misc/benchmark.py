import time
import csv

import numpy as np

import pymia.data.definition as defs
import pymia.data.extraction as extr


def main():
    hdf_file = '../examples/example-data/example-dataset.h5'
    file_root = '../examples/example-data'
    file_un_root = '../examples/example-data-uncompressed'
    file_np_root = '../examples/example-data-numpy'
    repetitions = 5

    results = {}
    repetitions = list(range(repetitions))
    for r in repetitions:
        print(r)
        for name, (extractor, indexing_strategy) in get_pairs(file_root, file_un_root, file_np_root).items():
            dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, None)

            start = time.time()
            for i, sample in enumerate(dataset):
                x = sample[defs.KEY_IMAGES]
            end = time.time()

            results.setdefault(name, []).append(end - start)

    with open('benchmark_results_w_numpy.csv', 'w', newline='') as f:
        names = ['repetition'] + list(results.keys())
        writer = csv.DictWriter(f, names)
        writer.writeheader()

        for r in repetitions:
            row = {'repetition': r, **{k: v[r] for k, v in results.items()}}
            writer.writerow(row)


def np_read(file_path, cat):
    return np.load(file_path + '.npy')


def get_pairs(file_root, uncompressed_file_root, numpy_file_root):
    data_extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES,))
    file_extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,), override_file_root=file_root)
    file_np_extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,), override_file_root=numpy_file_root, load_fn=np_read)
    file_un_extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,), override_file_root=uncompressed_file_root)

    slices = extr.SliceIndexing()
    patches = extr.PatchWiseIndexing((64, 64, 64))
    full = extr.EmptyIndexing()

    padding = (10, 10, 10)

    return {
        'dataset-slice': (data_extractor, slices),
        'file-slice': (file_extractor, slices),
        'file-un-slice': (file_un_extractor, slices),
        'file-np-slice': (file_np_extractor, slices),
        'dataset-patch': (extr.PadDataExtractor(padding, data_extractor), patches),
        'file-patch': (extr.PadDataExtractor(padding, file_extractor), patches),
        'file-un-patch': (extr.PadDataExtractor(padding, file_un_extractor), patches),
        'file-np-patch': (extr.PadDataExtractor(padding, file_np_extractor), patches),
        'dataset-full': (data_extractor, full),
        'file-full': (file_extractor, full),
        'file-un-full': (file_un_extractor, full),
        'file-np-full': (file_np_extractor, full),
    }


if __name__ == '__main__':
    main()
