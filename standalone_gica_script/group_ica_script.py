from ica import ica1
import pickle
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

# This script performs group ICA on a study's set of processed fMRI files
def load_and_flatten_files():
    print("Read the file paths from proc_fmri_datafiles.txt")
    with open('input_data/proc_fmri_datafiles.txt', 'r') as file:
        file_paths = file.read().split(',')

    print("Remove leading/trailing whitespace from file paths")
    file_paths = [path.strip() for path in file_paths]

    print("Initialize an empty list to store the flattened data")
    subject_matrices = []
    ix = 0

    print("Iterate over the file paths and load the NIfTI files")
    for path in file_paths:
        try:
            print("img = nib.load(path)")
            img = nib.load(path)

            print("data = img.get_fdata()")
            data = img.get_fdata()

            print("Calculate the total number of voxels")
            num_voxels = np.prod(data.shape[:-1])    

            print("num_voxels is ")
            print(num_voxels)

            print("Get the number of timepoints")
            num_timepoints = data.shape[-1]

            print("num_timepoints is ")
            print(num_timepoints)

            print("Create flattened matrix")
            flattened_matrix = data.reshape((num_voxels, num_timepoints))

            print("Insert flattened matrix into list")
            subject_matrices.insert(ix, np.array(flattened_matrix))
            ix += 1

        except:
            print(f"Error loading file: {path}")

    print("Concatenate all flattened subject matrices along the axis=1 axis")
    matrix = np.concatenate(subject_matrices, axis=1)


    print("Print the shape of the matrix")
    print("Matrix shape:", matrix.shape)
    print("matrix[0][0] is ")
    print(matrix[0][0])
    
    return matrix

def perform_group_ICA(matrix):
    n_components = 50
    X = matrix
    print(" 46 A,S,W = ica1(np.array(X), n_components)")
    A,S,W = ica1(np.array(X), n_components)

    print("X.shape #num voxels by num time points")
    print(X.shape) #num voxels by num time points

    print("A.shape  #num timepoints by num components; how active a component(network) is at each timepoint")
    print(A.shape)  #num timepoints by num components; how active a component(network) is at each timepoint

    print("S.shape  #num components by num voxels, how much a voxel contributes to a network (component)")
    print(S.shape)  #num components by num voxels, how much a voxel contributes to a network (component)


    print("W.shape  #mixing matrix")
    print(W.shape)

    return A,S,W


# This function writes the columns of A out as a nii file to visualize ICA on one subject
def visualize_one_sub(A, n_components):
    # Toggle these values to display slices of the brain
    xcoord=46
    ycoord=55
    zcoord=46
    volnum=0


    # These values are total # voxels in x,y,z dimension
    xdim=91
    ydim=109
    zdim=91
    total_vols=1200

    # Number of components found using ICA
    K = n_components

    # Create an empty 4D array to store all the 3D images
    image_stack = np.zeros((xdim, ydim, zdim, K))
    
    # For each component, reshape each col of A into a 3D image
    for ix in range(K):
        sm = A[:, ix]
        sm_pixels = np.array(sm).reshape(xdim, ydim, zdim)

        # Add the current 3D image to the stack
        image_stack[..., ix] = sm_pixels

    # Create a NIfTI image object from the stacked 4D array
    nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))

    # Save the NIfTI image to a file
    nifti_file = "SM.nii.gz"
    nib.save(nifti_img, nifti_file)

# This function saves the group matrix as a pickle
def pickle_group_mat(group_mat):
    filename = "matrix.pkl"
    with open(filename, "wb") as file:
        pickle.dump(group_mat, file)

# This function computes the error between the X and A matrix
def compute_error_matrix(X,M):
    diff = (X - M)
    error = np.mean(diff)
    return error

# This function performs PCA on the flattened data matrix X
def pca(X):
    	# compute the mean voxel intensity in the matrix X 
	mean = np.mean(X)
        
    	# subtract the mean from each voxel intensity
	zero_mean_mat = X - mean
        
   	# compute the covariance matrix of the zero mean image
	cov_mat = np.cov(zero_mean_mat.T)
        
    	# compute the eigenvalues and eigenvectors of the zeromean covariance matrix
	eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    	# sort the indices of the eigenvalues from largest to smallest
	idx = eigenvalues.argsort()[::-1]
	
    	# use the sorted indices of the eigenvalues to re arrange the eigenvectors
	eigenvectors = eigenvectors[:,idx]
        
   	# create a matrix from the ordered eigenvalues (should a subset of these be used?)
	kl_tx_mat = np.column_stack(eigenvectors)

    	# transform each datapoint by the     
	tx_data = np.dot(zero_mean_mat, kl_tx_mat)
	print("tx_data.shape: {}".format(tx_data.shape))


	return tx_data

# This function performs PCA whitening on an input matrix X
# Whiten the signal in X by centering and normalizing by the variance

def pca_whitening(X, n_comps):
    # Calculate the mean
    mean = np.mean(X)

    # Subtract the mean
    zero_mean_mat = X - mean

    # Compute the covariance matrix
    cov_mat = np.cov(zero_mean_mat.T)

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Get the top K eigenvectors, K=n_comps
    top_k_eigenvectors = sorted_eigenvectors[:, :n_comps]
    top_k_eigenvalues = sorted_eigenvalues[:n_comps]


    # Whiten the data (divide all entries in X by the variance)
    epsilon = 1e-5  # Small constant to avoid division by zero
    whitened_data = np.dot(zero_mean_mat, top_k_eigenvectors / np.sqrt(top_k_eigenvalues + epsilon))


    return whitened_data

def backward_project():
	# Transpose the Mixing Matrix W
	W_T = W.T

	# Separate Subject-Specific ICs
	Y = np.dot(W_T, X)  # Y has shape (n_components, num_timepoints)

	# Separate Subject-Specific Time Courses
	Y_T = Y.T  # Y_T has shape (num_timepoints, n_components)

	# Separate Subject-Specific ICs (Spatial Maps)
	A_T = A.T  # A_T has shape (n_components, num_voxels)

	# Subject-Specific ICs and TCs
	# Each row of Y_T represents the ICs for a subject
	subject_specific_ICs = Y_T  

	# Each row of A_T represents the TC for a subject
	subject_specific_TCs = A_T  




################################################################
print("group_matrix = load_and_flatten_files()")
group_matrix = load_and_flatten_files()

#print("A,S,W = perform_group_ICA(group_matrix)")
#A,S,W = perform_group_ICA(group_matrix)

n_comps = 50
#print("visualize_one_sub(A, n_components)")
#visualize_one_sub(A, n_comps)


#print("E = compute_error_matrix(group_matrix, A)")
#error = compute_error_matrix(group_matrix, A@S)

#print("error is {}".format(error))


print("pca_whitening(group_matrix)")
pca_img = pca_whitening(group_matrix, n_comps)

#print("pca_img = pca(group_matrix)")
#pca_img = pca(group_matrix)

print("pca_img matrix shape is {}".format(pca_img.shape))

# These values are total # voxels in x,y,z dimension
xdim=91
ydim=109
zdim=91
total_vols=pca_img.shape[1]
#total_vols=n_comps


print("image_stack = np.zeros((xdim, ydim, zdim, total_vols))")
image_stack = np.zeros((xdim, ydim, zdim, total_vols))




# For each component, reshape each col of A into a 3D image
for ix in range(total_vols):
    vol = pca_img[:, ix]
    vol_pixels = np.array(vol).reshape(xdim, ydim, zdim)

    image_stack[..., ix] = vol_pixels


print("nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))")
nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))

# Save the NIfTI image to a file
nifti_file = "pca_whitened_000800485677.nii.gz"

print("nib.save(nifti_img, {})".format(nifti_file))
nib.save(nifti_img, nifti_file)
