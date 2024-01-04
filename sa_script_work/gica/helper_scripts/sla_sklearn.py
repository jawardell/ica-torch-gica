import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from ica import pca_whiten


sub_id='000300655084'
func_filepath = '/Users/jwardell1/Desktop/000300655084/000300655084_rest.nii.gz'
mask_filepath = '/Users/jwardell1/Desktop/000300655084/groupmeanmask.nii'
output_dir = '/Users/jwardell1/Desktop/000300655084'
fmri_img = nib.load(func_filepath) 
mask = nib.load(mask_filepath) 
data = fmri_img.get_fdata() 
xdim, ydim, zdim, nvol = data.shape
nvox = xdim * ydim * zdim
x2d=data.reshape((nvox, nvol))

n_comp = 50
pca = PCA(n_components=n_comp, whiten=True)
pca_result = pca.fit_transform(x2d)
#pca_result, white, dewhite = pca_whiten(x2d.T, n_comp)

pca_volumes = np.zeros((xdim, ydim, zdim, n_comp))


for ix in range(n_comp):
    volume = pca_result[..., ix]
    sm_vox  = volume.reshape((xdim, ydim, zdim))
    pca_volumes[..., ix]  = sm_vox

nifti_filepath = "{}/{}_pca.nii.gz".format(output_dir, sub_id)
nifti_img = nib.Nifti1Image(pca_volumes, affine=np.eye(4))
nib.save(nifti_img, nifti_filepath)