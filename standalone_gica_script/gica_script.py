from ica import pca_whiten
from ica import ica1
import numpy as np
import sys
import os
import nibabel as nib
import torch

#### Excessive print statements are for debugging purposes and will be removed after debugging #######

if len(sys.argv) != 4:
    print("Usage: python gica_script.py sla_filepaths mask_filepath output_dir")
    print(sys.argv)
    sys.exit()

sla_filepaths = sys.argv[1]
mask_filepath = sys.argv[2]
output_dir = sys.argv[3]


if not os.path.isfile(sla_filepaths):
    print("Error: subject level analysis file not found.")
    sys.exit()


if not os.path.isfile(mask_filepath):
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


        print("flattened_matrix = np.array(torch.load(path))")
        flattened_matrix = np.array(torch.load(path))
	

        print("insert matrix {}/{} into list".format(ix, num_subs))
        subject_matrices.insert(ix, np.array(flattened_matrix))
        ix += 1

    except:
        print(f"Error loading file: {path}")

print("sl_concat = np.concatenate(subject_matrices, axis=1)")
sl_concat = np.concatenate(subject_matrices, axis=0)

print("dimensions should be numcomps * numsubs by numvoxels ")
print(" each column should contain the total number of brain voxels ")
print("Matrix shape:", sl_concat.shape)


##### run pca on concatenated matrix #####
print("n_pca_comps = 50")
n_pca_comps = 50

print("pca_res, white, dewhite = pca_whiten(sl_concat, n_pca_comps)")
pca_res, white, dewhite = pca_whiten(sl_concat, n_pca_comps)
print("pca_res.shape, white.shape, dewhite.shape")

##### run ica on pca result #####
print("n_ica_comps = 50")
n_ica_comps = 50

print("A,S,W = ica1(pca_res.T, n_ica_comps)")
A,S,W = ica1(pca_res.T, n_ica_comps)
print("A.shape, S.shape, W.shape is {}, {}, {}".format(A.shape, S.shape, W.shape))

##### save SMs as nifti #####
print("mask_img = nib.load(mask_filepath)")
mask_img = nib.load(mask_filepath)
mask_data = mask_img.get_fdata()
xdim, ydim, zdim = mask_data.shape

print("idx = np.where(mask_img.dataobj)")
idx = np.where(mask_img.dataobj)

print("image_stack = np.zeros((xdim, ydim, zdim, n_ica_comps))")
image_stack = np.zeros((xdim, ydim, zdim, n_ica_comps))

print("image_stack[*idx,:] = A")
image_stack[*idx,:] = A


'''
for ix in range(n_ica_comps):
        sm = A[:, ix]
        sm_pixels = np.array(sm).reshape(xdim, ydim, zdim)

        # Add the current 3D image to the stack
        image_stack[..., ix] = sm_pixels

'''
# Create a NIfTI image object from the stacked 4D array
print("nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))")
nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))


# Save the NIfTI image to a file
nifti_file = "{}/group_SM.nii.gz".format(output_dir)
print("group ica SMs filename is {}".format(nifti_file))

print("nib.save(nifti_img, nifti_file)")
nib.save(nifti_img, nifti_file)


####### save data as torch files #####

