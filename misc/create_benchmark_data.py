import os
import shutil
import glob

import numpy as np

import SimpleITK as sitk


in_dir = '../examples/example-data'
out_dir_un = '../examples/example-data-uncompressed'
out_dir_np = '../examples/example-data-numpy'

if os.path.isdir(out_dir_un):
    shutil.rmtree(out_dir_un)
os.makedirs(out_dir_un)

if os.path.isdir(out_dir_np):
    shutil.rmtree(out_dir_np)
os.makedirs(out_dir_np)

mha_files = glob.glob(in_dir + '/**/*.mha')
for mha_file in mha_files:
    img = sitk.ReadImage(mha_file)

    out_un_mha = mha_file.replace(in_dir, out_dir_un)
    os.makedirs(os.path.dirname(out_un_mha), exist_ok=True)
    sitk.WriteImage(img, out_un_mha, False)

    out_np_mha = mha_file.replace(in_dir, out_dir_np)
    os.makedirs(os.path.dirname(out_np_mha), exist_ok=True)
    np_array = sitk.GetArrayFromImage(img)
    np.save(out_np_mha, np_array)

