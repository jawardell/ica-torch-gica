#this is brads code, to use, run   pip uninstall ica ...
#from ica_torch.ica_torch import ica1 


# this is alvero's code, to use, run   pip install ica ...
from ica import ica1


import numpy as np
import sys
import os
import nibabel as nib
import torch
from sklearn import preprocessing

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


with open(sla_filepaths, 'r') as file:
        file_paths = file.read().split(',')

file_paths = [path.strip() for path in file_paths]

subject_matrices = []
ix = 0
num_subs = len(file_paths)



for path in file_paths:
    try:
        if not os.path.isfile(sla_filepaths):
            print("Error: subject level pca nii not found.")
            sys.exit()


        flattened_matrix = np.array(torch.load(path))
	

        subject_matrices.insert(ix, np.array(flattened_matrix))
        ix += 1

    except:
        print(f"Error loading file: {path}")

sl_concat = np.concatenate(subject_matrices, axis=0)



##### run ica on concatenated matrix #####
n_ica_comps = 100

A,S,W = ica1(torch.from_numpy(sl_concat), n_ica_comps)


print("A.shape {}".format(A.shape))
print("S.shape {}".format(S.shape))
print("W.shape {}".format(W.shape))

np.save('A.npy', A)
np.save('S.npy', S)
np.save('W.npy', W)



##### save SMs as nifti #####
mask_img = nib.load(mask_filepath)
mask_data = mask_img.get_fdata()
xdim, ydim, zdim = mask_data.shape

idx = np.where(mask_img.dataobj)

image_stack = np.zeros((xdim, ydim, zdim, n_ica_comps))
x,y = A.shape
A_scaled = np.zeros((x,y))



# normalize each component's intensity values
'''
for ix in range(n_ica_comps):
	comp = A[:,ix]
	#scaled_comp = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(comp.reshape(-1,1))
	scaled_comp = preprocessing.StandardScaler().fit_transform(torch.flatten(comp))
	print(scaled_comp.shape,x,A_scaled.shape)
	A_scaled[:,ix] = scaled_comp.reshape(x)
'''

#image_stack[*idx,:] = preprocessing.StandardScaler().fit_transform(A)#A_scaled 



image_stack[*idx,:] = preprocessing.StandardScaler().fit_transform(S.T)


#trying to orient result to match mask and prep data might not work though 
nifti_img = nib.Nifti1Image(image_stack, affine=mask_img.get_qform()) 


nifti_img.header.set_qform(mask_img.header.get_qform(), code=mask_img.get_qform('code')[1])  # Set the qform from the mask code 4 means MNI space
nifti_img.header.set_sform(mask_img.header.get_sform(), code=mask_img.get_qform('code')[1])  # Set the sform from the mask code 4 means MNI space
nifti_img.header.set_xyzt_units(xyz='mm') # Set the xyz units to mm



# Save the NIfTI image to a file
nifti_file = "{}/group_SM.nii.gz".format(output_dir)

nib.save(nifti_img, nifti_file)
