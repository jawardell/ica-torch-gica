from ica import pca_whiten
import numpy as np
import nibabel as nib
import sys
import os
import torch

# This is a script that can compute each subject's PCA whitened data in parallel
# The PCA whitened data is num_comps by num_voxels and is written out as a binary file




#### Parse Script Arguments ####
#  set up arguments for script
if len(sys.argv) != 5:
    print("Usage: python subject_level_pca.py subject_ID fMRI_filepath output_path mask_filepath")
    print(sys.argv)
    sys.exit()

sub_id = sys.argv[1]
func_filepath = sys.argv[2]
output_dir = sys.argv[3]
mask_filepath = sys.argv[4]

if not os.path.isfile(func_filepath):
    print("Error: fMRI file not found.")
    sys.exit()

if not os.path.isfile(mask_filepath):
    print("Error: mask file not found.")
    sys.exit()

if not os.path.isdir(output_dir):
    print("Error: output dir not found.")
    sys.exit()



#### Load Neuroimage Data ####

# load fMRI data
fmri_img = nib.load(func_filepath) 

# load brain mask
mask = nib.load(mask_filepath) 

# capture mask indices of non-zero elements
idx = np.where(mask.dataobj) 

# this is the data of a subject with the last dimension being the total time duration
data = fmri_img.get_fdata() 



# voxel by time data to apply the PCA to
data2pca = data[*idx,:]

# number of components to retain
n_comps = 120

# ... apply PCA here to the appropriately transposed data to obtain pca_result (n_comps x voxels)
pca_result, white, dewhite = pca_whiten(data2pca.T, n_comps)







#### Save PCA Result to Cluster ####

##### Save data as NIfTI (Optional) #####
# create empty 4d array
pca_volumes = np.zeros((data.shape[0], data.shape[1], data.shape[2],  n_comps))


# insert non-zero voxel intensities into 4d array 
pca_volumes[*idx,:] = pca_result.T


# create a nibabel image using the 4d array of pca data
nifti_img = nib.Nifti1Image(pca_volumes, affine=np.eye(4))

# make the save file string
nifti_file = "{}/{}_pca.nii.gz".format(output_dir, sub_id)
print("pca nifti filename is {}".format(nifti_file))

# write the nifti to the disk
nib.save(nifti_img, nifti_file)


##### Save data as binary file for GICA #####
# make the save file string
filename = "{}/{}_pca.torch".format(output_dir, sub_id)
print("torch binary filename is {}".format(filename))

# write pca matrix to disk (ncomps x voxels)
nvox = data.shape[0] * data.shape[1] * data.shape[2]
torch.save(pca_result, filename)
