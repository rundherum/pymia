import os
import shutil
import glob

import SimpleITK as sitk


in_dir = '../examples/example-data'
out_dir = '../examples/example-data-uncompressed'

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

mha_files = glob.glob(in_dir + '/**/*.mha')
for mha_file in mha_files:
    out_mha = mha_file.replace(in_dir, out_dir)
    os.makedirs(os.path.dirname(out_mha), exist_ok=True)  # create dir if not yet existing
    sitk.WriteImage(sitk.ReadImage(mha_file), out_mha, False)

nii_files = glob.glob(in_dir + '/**/*.nii.gz')
for nii_file in nii_files:
    out_nii = nii_file.replace(in_dir, out_dir)[:-len('.gz')]
    os.makedirs(os.path.dirname(out_nii), exist_ok=True)  # create dir if not yet existing
    sitk.WriteImage(sitk.ReadImage(nii_file), out_nii, False)
