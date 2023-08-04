from ica import pca_whiten
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import sys
import os



def load_and_flatten_nifti(func_filepath):
    """ Loads the functional data from the filepath 
    and flattens it into a 2D matrix of time by voxels.
    :param func_filepath: Full filepath to an fMRI file on the cluster.
    :type func_filepath: string
    :return: A 2D numpy float array of the masked fMRI data
    :rtype: numpy.array(dtype=float)
    """
    print("Load and flatten NIfTI file")
    print("img = nib.load(path)")
    img = nib.load(func_filepath)

    print("data = img.get_fdata()")
    data = img.get_fdata()

    print("data.shape : {}".format(data.shape))

    print("Calculate the total number of voxels")
    num_voxels = np.prod(data.shape[:-1])    

    print("num_voxels is ")
    print(num_voxels)

    print("Get the number of timepoints")
    num_timepoints = data.shape[-1]

    print("num_timepoints is ")
    print(num_timepoints)

    print("flattened_matrix = data.reshape((num_timepoints, num_voxels))")
    
    flattened_matrix = data.reshape((num_voxels, num_timepoints))

    print("data.shape : {}".format(data.shape))
    print("flattened_matrix.shape : {}".format(flattened_matrix.shape))
    return (data, flattened_matrix)


def save_mat_as_text(X, outpath, sub_id, prefix):
    """ Writes the matrix X out as a .txt file to the 
    specified output directory.
    :param X: Flattened fMRI data matrix
    :type X: numpy.array(dtype=float)
    :param outpath: Full path to output directory to write 
        data matrix out to. 
        Do not include / at the end of the directory name.
    :type outpath: string
    :param sub_id: Subject ID
    :type sub_id: string
    :param prefix: Prefix used to form filename.
    :type prefix: string
    """
    filename = "{}/{}_{}".format(outpath, sub_id, prefix)
    print("np.savetxt(filename, X)")
    np.savetxt(filename, X)


def save_data_as_nifti(X, outpath, sub_id, prefix, num_vols, xdim, ydim, zdim):
    """ Writes the matrix X out as a .txt file to the 
    specified output directory.
    :param X: Flattened fMRI data matrix
    :type X: numpy.array(dtype=float)
    :param outpath: Full path to output directory to write 
        data matrix out to. 
        Do not include / at the end of the directory name.
    :type outpath: string
    :param sub_id: Subject ID
    :type sub_id: string
    :param prefix: Prefix used to form filename.
    :type prefix: string
    :param num_vols: Number of volumes, or timepoints, in fMRI file
    :type num_vols: integer
    :param xdim: Number of voxels in the x dimension of fMRI file
    :type xdim: integer
    :param ydim: Number of voxels in the y dimension of fMRI file
    :type ydim: integer
    :param zdim: Number of voxels in the z dimension of fMRI file
    :type zdim: integer
    """

    # Create an empty 4D array to store all the 3D images
    image_stack = np.zeros((xdim, ydim, zdim, num_vols))

    for ix in range(num_vols):
        volume = X[ix, ...]
        print("X.shape {}".format(X.shape))
        print("volume.shape {}".format(volume.shape))

        volume_voxels = np.array(volume).reshape(xdim, ydim, zdim)

        # Add the current 3D image to the stack
        image_stack[..., ix] = volume_voxels

    # Create a NIfTI image object from the stacked 4D array
    nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))

    # Save the NIfTI image to a file
    nifti_file = "{}/{}_{}.nii.gz".format(outpath, sub_id, prefix)
    nib.save(nifti_img, nifti_file)

def flatten_nifti(sub_id, func_filepath, mask_filepath, output_dir):
    """ This is the entry point for the script.
    The script should be called passing these parameters as arguments. 
    Loads and flattens fMRI file, performs PCA whitening on it, 
    and writes it out to a .txt file.
    :param sub_id: subject's unique identifier
    :type sub_id: string
    :param func_file: path to subject's fMRI data
    :type func_file: string
    :param mask_file: path to group mask file
    :type mask_file: string
    :param output_dir: path to output directory
    :type output_dir: string
    """
    num_vols=162
    xdim=91
    ydim=109
    zdim=91

    num_components = 150

    print("img = nib.load(groupmask_file)")
    mask_img = nib.load(mask_filepath)
    
    print("data = img.get_fdata")
    mask_data = mask_img.get_fdata()

    print("boolean_mask = np.asarray(mask_data, dtype=bool)")
    boolean_mask = np.asarray(mask_data, dtype=bool)
    
    print("func_f_m = load_and_flatten_nifti(func_filepath)")
    # func --> fMRI data
    # f --> data has been flattened
    # m --> data has been masked
    data, func_f_m = load_and_flatten_nifti(func_filepath)
    num_voxels = func_f_m.shape[0]

    # perform PCA whitening on only the brain voxels

    print("num_brain_voxels = boolean_mask.sum()")
    num_brain_voxels = boolean_mask.sum()

    print("brain_voxels = data[boolean_mask].reshape(num_vols, num_brain_voxels)")
    brain_voxels = data[boolean_mask].reshape(num_vols, num_brain_voxels)

    print("x_white, white, dewhite = pca_whiten(brain_voxels)")
    x_white, white, dewhite = pca_whiten(brain_voxels, num_components)



    image_stack = np.zeros((xdim, ydim, zdim, num_components))
    image_stack[boolean_mask, ...] = x_white.T

    # func --> fMRI data
    # f --> data has been flattened
    # m --> data has been masked
    # m --> data has been whitened
    # pca --> pca has been applied to the data
    print("image_stack.shape {}".format(image_stack.shape))
    func_f_m_w_pca = image_stack.reshape(num_components, num_voxels)
    print("func_f_m_w_pca.shape : {}".format(func_f_m_w_pca.shape))
    
    prefix = "func_f_m_w_pca"



    # save whitened data as nifti to visualize whitening
    save_data_as_nifti(func_f_m_w_pca, output_dir, sub_id, prefix, num_components, xdim, ydim, zdim)
    save_mat_as_text(func_f_m_w_pca, output_dir, sub_id, prefix)

    


if __name__ == "__main__":
    if len(sys.argv) != 5:
            print("Usage: python flatten_nifti.py subject_ID fMRI_filepath brainmask_filepath output_path")
            print(sys.argv)
            sys.exit()

    sub_id = sys.argv[1]
    func_filepath = sys.argv[2]
    mask_filepath = sys.argv[3]
    output_dir = sys.argv[4]

    if not os.path.isfile(func_filepath):
        print("Error: fMRI file not found.")
        sys.exit()
    
    if not os.path.isfile(mask_filepath):
        print("Error: brain mask file not found.")
        sys.exit()

    if not os.path.isdir(output_dir):
        print("Error: output dir not found.")
        sys.exit()

    flatten_nifti(sub_id, func_filepath, mask_filepath, output_dir)


