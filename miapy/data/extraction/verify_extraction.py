import os

import numpy as np
import SimpleITK as sitk
import torch.utils.data as data

import miapy.data.extraction.sample as smpl
import miapy.data.extraction.dataset as d
import miapy.data.extraction.reader as r
import miapy.data.creation.writer as wr
import miapy.data.extraction.indexing as idx
import miapy.data.extraction.extractor as extr
import libs.util.filehelper as fh


def main():
    in_file = '../../in/datasets/ds2.h5'
    reader = r.Hdf5Reader(in_file, 'data/sequences')
    reader.open()

    extractors = [extr.FilesExtractor(), extr.NamesExtractor(), extr.DataExtractor()]
    ds = d.ParametrizableDataset(reader, idx.SliceIndexing(), extractors)

    entry = 'temp/nonblack_indices'
    if reader.has(entry):
        indices = reader.read(entry).tolist()
    else:
        indices = smpl.select_indices(ds, smpl.NonBlackSelection())
        # need to close the reader first
        reader.close()
        with wr.Hdf5Writer(in_file) as writer:
            writer.write(entry, indices)
        reader.open()

    loader = data.DataLoader(ds, sampler=smpl.SubsetSequentialSampler(indices))

    out_dir = '../../out/cavity/test/'
    print(len(ds))

    seq_names = list(ds[0]['sequence_names'])
    t1_index = seq_names.index('t1')
    prev_t1 = None
    content = []
    for batch in loader:
        for index in range(loader.batch_size):
            curr_t1 = batch['sequence_files'][t1_index][index]
            if prev_t1 is None:
                prev_t1 = curr_t1
            if prev_t1 != curr_t1:
                t1_file = os.path.join(out_dir, prev_t1)
                fh.create_dir_if_not_exists(t1_file, is_file=True)
                np_content = np.asarray(content)
                sitk.WriteImage(sitk.GetImageFromArray(np_content), t1_file)
                content = []
            content.append(batch['images'][index, ..., t1_index].numpy())
            prev_t1 = curr_t1

    reader.close()


if __name__ == '__main__':
    main()
