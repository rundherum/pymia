import time
import csv

import pymia.data.definition as defs
import pymia.data.extraction as extr


def main():
    hdf_file = '../examples/example-data/example-dataset.h5'
    file_root = '../examples/example-data-uncompressed'
    repetitions = 5

    results = {}
    repetitions = list(range(repetitions))
    for r in repetitions:
        print(r)
        for name, (extractor, indexing_strategy) in get_pairs(file_root).items():
            dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, None)

            start = time.time()
            for i, sample in enumerate(dataset):
                x = sample[defs.KEY_IMAGES]
            end = time.time()

            results.setdefault(name, []).append(end - start)

    with open('benchmark_results.csv', 'w', newline='') as f:
        names = ['repetition'] + list(results.keys())
        writer = csv.DictWriter(f, names)
        writer.writeheader()

        for r in repetitions:
            row = {'repetition': r, **{k: v[r] for k, v in results.items()}}
            writer.writerow(row)


def get_pairs(uncompressed_file_root):
    data_extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES,))
    file_extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,))
    file_un_extractor = extr.FilesystemDataExtractor(categories=(defs.KEY_IMAGES,), override_file_root=uncompressed_file_root)

    slices = extr.SliceIndexing()
    patches = extr.PatchWiseIndexing((64, 64, 64))
    full = extr.EmptyIndexing()

    padding = (10, 10, 10)

    return {
        'dataset-slice': (data_extractor, slices),
        'file-slice': (file_extractor, slices),
        'file-un-slice': (file_un_extractor, slices),
        'dataset-patch': (extr.PadDataExtractor(padding, data_extractor), patches),
        'file-patch': (extr.PadDataExtractor(padding, file_extractor), patches),
        'file-un-patch': (extr.PadDataExtractor(padding, file_un_extractor), patches),
        'dataset-full': (data_extractor, full),
        'file-full': (file_extractor, full),
        'file-un-full': (file_un_extractor, full),
    }


if __name__ == '__main__':
    main()
