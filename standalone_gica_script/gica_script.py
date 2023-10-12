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




import numpy as np
import numpy.linalg as la

def icatb_multi_fixed_ICA_R_Cor(mixedsig, refsig, loopnum, threshold, i_Rou, i_Uk, i_Gama, epsilon):
    # Removes the mean of mixedsig and refsig.
    mixedsigr = mixedsig - np.mean(mixedsig, axis=1, keepdims=True)
    refsig = refsig - np.mean(refsig, axis=1, keepdims=True)

    # Pre-whitening mixedsig and refsig
    Rxx = np.cov(mixedsigr)
    AV, B = la.eig(Rxx)
    PreWhiten = np.dot(np.linalg.inv(np.sqrt(B)), AV.T)
    DepreWhiten = np.dot(AV, np.sqrt(B))
    X = np.dot(PreWhiten, mixedsigr)

    r = refsig / np.std(refsig, axis=1, ddof=0, keepdims=True)
    
    # Parameters Setting
    dim, numsamp = mixedsig.shape
    dimr, numsampr = r.shape
    threshold = threshold * np.ones(dimr)

    Rou = i_Rou * np.ones(dimr)
    Uk = i_Uk * np.ones(dimr)
    Gama = i_Gama * np.ones(dimr)
    
    v = np.random.randn(dimr, numsamp)
    W = np.zeros((dim, dimr))
    Wold = W.copy()
    y = np.zeros((dimr, numsamp)

    for j in range(loopnum):
        W = np.dot(X, np.dot(icatb_Gdfun(np.dot(W.T, X)).T, np.diag(Rou)) / numsamp + 0.5 * np.dot(X, np.dot(r.T, np.diag(Uk)) / numsamp
        W = W / la.norm(W, axis=0)
        y = np.dot(W.T, X)

        for k in range(dimr):
            if Uk[k] + Gama[k] * icatb_gw_yr_Cor(np.dot(W[:, k].T, X), r[k], numsamp, threshold[k]) > 0:
                Uk[k] = Uk[k] + Gama[k] * icatb_gw_yr_Cor(np.dot(W[:, k].T, X), r[k], numsamp, threshold[k])
            else:
                Uk[k] = 0

        Rou = np.mean(icatb_Gfun(y), axis=1) - np.mean(icatb_Gfun(v), axis=1)

        W = W.T
        W1 = la.sqrtm(np.dot(W, W.T))
        W2 = la.inv(W1)
        W = np.dot(W2, W)
        W = W.T

        minAbsW = np.min(np.abs(np.diag(np.dot(W.T, Wold)))
        if 1 - minAbsW < epsilon:
            print('computed (', j, 'steps)')
            break

        Wold = W

    t_fixed_Cor = None  # You can replace this with timing code
    W = np.dot(W.T, PreWhiten)
    out_M = y
    return out_M, W

# You'll need to define or replace icatb_Gdfun, icatb_gw_yr_Cor, and icatb_Gfun functions






########################################################################
# CALL FUNCTIONS
########################################################################

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
