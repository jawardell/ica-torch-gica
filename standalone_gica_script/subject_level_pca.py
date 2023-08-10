from ica import pca_whiten
import numpy as np
import nibabel as nib
import sys
import os
import torch
import torch.linalg as tla

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
print("fmri_img = nib.load(func_filepath)")
fmri_img = nib.load(func_filepath) 

# load brain mask
print("mask = nib.load(mask_filepath)")
mask = nib.load(mask_filepath) 

# capture mask indices of non-zero elements
print("idx = np.where(mask.dataobj)")
idx = np.where(mask.dataobj) 

# this is the data of a subject with the last dimension being the total time duration
print("data = fmri_img.get_fdata()")
data = fmri_img.get_fdata() 

# voxel by time data to apply the PCA to
print("data2pca = data[*idx,:]")
data2pca = data[*idx,:]
print("data2pca.T.shape (observations by variables) {}".format(data2pca.T.shape))


# number of components to retain
n_comps = data.shape[3]
print("n_comps = {}".format(n_comps))

# ... apply PCA here to the appropriately transposed data to obtain pca_result (n_comps x voxels)
print("pca_result, white, dewhite = pca_whiten(data2pca.T, n_comps)")
pca_result, white, dewhite = pca_whiten(data2pca.T, n_comps)








#### Inspect PCA result #### 

# check covariance matrix of whitening matrix
xdim,ydim,zdim,vols = data.shape
nvx = xdim*ydim*zdim
data_flat = data.reshape((nvx, vols)) # vox by timepts
x_demean = data_flat - data_flat.mean(axis=1).reshape((-1, 1))

n_tpts, n_vox  = white.shape

white_cov = (white @ white.T)/(n_tpts - 1)
print("white_cov {}".format(white_cov))

print("\n\n\n\n")

# try reconstructing original data from dewhite matrix
recon = np.dot(dewhite, pca_result)
recon_vols = np.zeros((xdim,ydim,zdim,vols))
recon_vols[*idx,:] = recon.T
nifti_file = "{}/{}_recon.nii.gz".format(output_dir, sub_id)
print("recon file is: {}".format(nifti_file))
nifti_img = nib.Nifti1Image(recon_vols, affine=np.eye(4))
print("nib.save(nifti_img, nifti_file)")
nib.save(nifti_img, nifti_file)

rcn = recon_vols.reshape((nvx, vols))
mean_err = np.mean(x_demean - rcn)
print("x_demean is {}".format(x_demean))
print("recon is {}".format(recon))
print("mean error is {}".format(mean_err))



# visualize demeaned data as nifti file
mean = np.mean(data)
print("mean is {}".format(mean))
x_demean = data - mean
nifti_file = "{}/{}_demeaned.nii.gz".format(output_dir, sub_id)
print("demeaned file is: {}".format(nifti_file))
nifti_img = nib.Nifti1Image(x_demean, affine=np.eye(4))
nib.save(nifti_img, nifti_file)





#### Save PCA Result to Cluster ####

##### Save data as NIfTI (Optional) #####
# create empty 4d array
print("pca_volumes = np.zeros((data.shape[0], data.shape[1], data.shape[2],  n_comps))")
pca_volumes = np.zeros((data.shape[0], data.shape[1], data.shape[2],  n_comps))


# insert non-zero voxel intensities into 4d array 
print("pca_volumes[*idx,:] = pca_result.T")
pca_volumes[*idx,:] = pca_result.T


# create a nibabel image using the 4d array of pca data
print("nifti_img = nib.Nifti1Image(pca_volumes, affine=np.eye(4))")
nifti_img = nib.Nifti1Image(pca_volumes, affine=np.eye(4))

# make the save file string
nifti_file = "{}/{}_pca.nii.gz".format(output_dir, sub_id)
print("pca nifti filename is {}".format(nifti_file))

# write the nifti to the disk
print("nib.save(nifti_img, nifti_file)")
nib.save(nifti_img, nifti_file)


##### Save data as binary file for GICA #####
# make the save file string
filename = "{}/{}_pca.torch".format(output_dir, sub_id)
print("torch binary filename is {}".format(filename))

# write pca matrix to disk (ncomps x voxels)
print("torch.save(pca_result, filename)")
torch.save(pca_result, filename)