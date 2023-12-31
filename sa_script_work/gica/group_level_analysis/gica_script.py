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
n_ica_comps = 100

def load_subject_matrices(file_paths):
	file_paths = [path.strip() for path in file_paths]

	subject_matrices = []
	ix = 0
	num_subs = len(file_paths)



	for path in file_paths:
		try:
			if not os.path.isfile(sla_filepaths):
				print(f"Error: subject level pca nii not found. {path}")
				#sys.exit()


			flattened_matrix = np.array(torch.load(path))
	
			subject_matrices.insert(ix, np.array(flattened_matrix))
			ix += 1

		except:
			print(f"Error loading file: {path}")

	sl_concat = np.concatenate(subject_matrices, axis=0)
	return sl_concat


def perform_ica(sl_concat):
	##### run ica on concatenated matrix #####
	

	A,S,W = ica1(torch.from_numpy(sl_concat), n_ica_comps)


	print("A.shape {}".format(A.shape))
	print("S.shape {}".format(S.shape))
	print("W.shape {}".format(W.shape))

	np.save('A.npy', A)
	np.save('S.npy', S)
	np.save('W.npy', W)

	return S



def save_as_nifti(S):
	##### save SMs as nifti #####
	mask_img = nib.load(mask_filepath)
	mask_data = mask_img.get_fdata()
	xdim, ydim, zdim = mask_data.shape

	idx = np.where(mask_img.dataobj)

	image_stack = np.zeros((xdim, ydim, zdim, n_ica_comps))
	
	print(type(S))
	# rebuild the 4D tensor of brain voxels
	image_stack[*idx,:] = S.T



	#trying to orient result to match mask and prep data might not work though 
	nifti_img = nib.Nifti1Image(image_stack, affine=mask_img.get_qform()) 


	nifti_img.header.set_qform(mask_img.header.get_qform(), code=mask_img.get_qform('code')[1])  # Set the qform from the mask code 4 means MNI space
	nifti_img.header.set_sform(mask_img.header.get_sform(), code=mask_img.get_qform('code')[1])  # Set the sform from the mask code 4 means MNI space
	nifti_img.header.set_xyzt_units(xyz='mm') # Set the xyz units to mm



	# Save the NIfTI image to a file
	nifti_file = "{}/group_SM.nii.gz".format(output_dir)

	nib.save(nifti_img, nifti_file)


def visual_normalize(S):
	# perform zscore normalization on the ica components
	#~~~~~~~~~~~~~> Should we leave this to brainbow? 
	S = preprocessing.StandardScaler().fit_transform(S.T)

	# normalize each ICA components' voxels by the max absolute intensity value
	#~~~~~~~~~~~~~> Should we leave this to brainbow? 
	S = (np.diag(1/np.abs(S.T).max(axis=1))@S.T).astype('float32')	
	midx = np.argmax(np.abs(S), axis=1)
	signs = np.diag(S[np.arange(S.shape[0]), midx])
	print(type)
	S = signs @ S
	return S




if len(sys.argv) != 4:
    print("Usage: python gica_script.py sla_filepaths mask_filepath output_dir")
    print(sys.argv)
    sys.exit()

sla_filepaths = sys.argv[1]
mask_filepath = sys.argv[2]
output_dir = sys.argv[3]
file_paths = []

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



from memory_profiler import profile

@profile
def profiled_code():    
	sl_concat = load_subject_matrices(file_paths)
	S = perform_ica(sl_concat)
	S_norm = visual_normalize(S)
	save_as_nifti(S_norm)

profiled_code()



#import cProfile
#cProfile.run('profiled_code()')
