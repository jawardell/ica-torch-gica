from ica import pca_whiten
from ica import ica1
import numpy as np
import sys
import os
import nibabel as nib
import torch



if len(sys.argv) != 2:
    print("Usage: python gica_script.py sla_filepaths")
    print(sys.argv)
    sys.exit()

sla_filepaths = sys.argv[1]
output_dir = sys.argv[2]


if not os.path.isfile(sla_filepaths):
    print("Error: subject level analysis file not found.")
    sys.exit()

if not os.path.isdir(output_dir):
    print("Error: output dir not found.")
    sys.exit()


print("Read the file paths from sla_filepaths txt file")
with open(sla_filepaths, 'r') as file:
        file_paths = file.read().split(',')

print("Remove leading/trailing whitespace from file paths")
file_paths = [path.strip() for path in file_paths]

print("Initialize an empty list to store the flattened data")
subject_matrices = []
ix = 0
num_subs = len(file_paths)



for path in file_paths:
    try:
        if not os.path.isfile(sla_filepaths):
            print("Error: subject level pca nii not found.")
            sys.exit()

        
        flattened_matrix = torch.load(path).numpy()

        subject_matrices.insert(ix, np.array(flattened_matrix))
        ix += 1

    except:
        print(f"Error loading file: {path}")

sl_concat = np.concatenate(subject_matrices, axis=1)


print("Matrix shape:", sl_concat.shape)


##### run pca on concatenated matrix #####
n_pca_comps = 150
pca_res = pca_whiten(sl_concat, n_pca_comps)


##### run ica on pca result #####
n_ica_comps = 50
A,S,W = ica1(pca_res, n_ica_comps)

##### save SMs as nifti #####
xdim=91
ydim=109
zdim=91
image_stack = np.zeros((xdim, ydim, zdim, n_ica_comps))

for ix in range(n_ica_comps):
        sm = A[:, ix]
        sm_pixels = np.array(sm).reshape(xdim, ydim, zdim)

        # Add the current 3D image to the stack
        image_stack[..., ix] = sm_pixels


# Create a NIfTI image object from the stacked 4D array
nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))


# Save the NIfTI image to a file
nifti_file = "{}/group_SM.nii.gz".format(output_dir)
nib.save(nifti_img, nifti_file)
