from ica import ica1
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

print("Read the file paths from proc_fmri_datafiles.txt")
with open('proc_fmri_datafiles.txt', 'r') as file:
    file_paths = file.read().split(',')

print("Remove leading/trailing whitespace from file paths")
file_paths = [path.strip() for path in file_paths]

print("Initialize an empty list to store the flattened data")
flattened_data = []

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
    except:
        print(f"Error loading file: {path}")

print("Convert the flattened data to a 2D matrix")
matrix = np.array(flattened_matrix)

print("Print the shape of the matrix")
print("Matrix shape:", matrix.shape)
print("matrix[0][0] is ")
print(matrix[0][0])

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




xcoord=46
ycoord=55
zcoord=46
volnum=0

xdim=91
ydim=109
zdim=91
total_vols=1200

sample_timepoint=1

K = n_components

# Create an empty 4D array to store all the 3D images
image_stack = np.zeros((xdim, ydim, zdim, K))

for ix in range(K):
    sm = A[:, ix]
    sm_pixels = np.array(sm).reshape(xdim, ydim, zdim)

    # Add the current 3D image to the stack
    image_stack[..., ix] = sm_pixels

# Create a NIfTI image object from the stacked 4D array
nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))

# Save the NIfTI image to a file
nifti_file = "3d_plots.nii.gz"
nib.save(nifti_img, nifti_file)

