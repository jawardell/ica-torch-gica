from ica import ica1


import numpy as np
import sys
import os
import torch
import scipy.io


n_pca_comps = 100

def load_subject_matrices(file_paths):
	file_paths = [path.strip() for path in file_paths]

	subject_matrices = []
	ix = 0
	num_subs = len(file_paths)



	for path in file_paths:
		try:
			if not os.path.isfile(sla_filepaths):
				print(f"Error: subject level pca torch file not found. {path}")
				#sys.exit()


			flattened_matrix = np.array(torch.load(path))
	
			subject_matrices.insert(ix, np.array(flattened_matrix))
			ix += 1

		except:
			print(f"Error loading file: {path}")

	sl_concat = np.concatenate(subject_matrices, axis=0)
	return sl_concat


SAVE_DIR = "/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work"

if len(sys.argv) != 2:
	print("Usage: python reduce_slconcat.py sla_filepaths")
	print(sys.argv)
	sys.exit()

sla_filepaths = sys.argv[1]

if not os.path.isfile(sla_filepaths):
	print("Error: subjec level analysis file not found.")
	sys.exit()

with open(sla_filepaths, 'r') as file: 
	file_paths = file.read().split(',')

sl_concat = load_subject_matrices(file_paths)
print(f'sl_concat shape: {sl_concat.shape}')


x_white, white, dewhite = pca_whiten(sl_concat, n_pca_comps, verbose=True)


print(f'x_white shape: {x_white.shape}')

matlab_file = '/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/fbirn_sla_pca.mat'
scipy.io.savemat(matlab_file, {'data': x_white})

