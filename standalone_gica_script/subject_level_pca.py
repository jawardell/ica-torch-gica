from ica import pca_whiten
import numpy as np
import nibabel as nib
import sys
import os
import torch

# This is a script that can compute each subject's PCA whitened data in parallel
# The PCA whitened data is num_timepoints by num_voxels and is written out as a txt file

def save_mat_as_bin(X, outpath, sub_id, prefix):
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
    filename = "{}/{}_{}.t".format(outpath, sub_id, prefix)
    print("torch.save(X, filename)")
    torch.save(X, filename)


def save_data_as_nifti(X, outpath, sub_id, prefix, fmri_dimensions, n_comps):
    """ Writes the matrix X out as a NIfTI file to the 
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
    :param fmri_dimensions: A 4-tuple containing #vols, xdim, ydim, zdim
	of the input fMRI NIfTI file
    :type fmri_dimensions: tuple of integers
    :return: none
    :rtype: none
    """
    xdim = fmri_dimensions[0]
    ydim = fmri_dimensions[1]
    zdim = fmri_dimensions[2]
    num_vols = n_comps


    # Create an empty 4D array to store all the 3D images
    print("image_stack = np.zeros((xdim, ydim, zdim, num_vols))")
    image_stack = np.zeros((xdim, ydim, zdim, num_vols))

    print("image_stack.shape".format(image_stack.shape))
    print("X.shape".format(X.shape))
    
    for ix in range(num_vols):
        volume = X[ix, ...]

        volume_voxels = np.array(volume).reshape(xdim, ydim, zdim)

        # Add the current 3D image to the stack
        image_stack[..., ix] = volume_voxels

    # Create a NIfTI image object from the stacked 4D array
    print("nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))")
    nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))

    # Save the NIfTI image to a file
    nifti_file = "{}/{}_{}.nii.gz".format(outpath, sub_id, prefix)
    print("nifti_file = {}".format(nifti_file))
    
    nib.save(nifti_img, nifti_file)


def flatten_nifti(fmri_data): 
   """ This function takes a 4D image of fMRI data 
	flattens it into a 2D matrix of time by voxels
	and returns it
	:param fmri_data: A 4D matrix of fMRI data
	:type fmri_img: numpy.array(dtype=float)
	:return: A 2D matrix of time by voxels 
	:rtype: numpy.array(dtype=float)
   """
   print("Calculate the total number of voxels")
   num_voxels = np.prod(fmri_data.shape[:-1])    

   print("num_voxels is ")
   print(num_voxels)

   print("Get the number of timepoints")
   num_timepoints = fmri_data.shape[-1]

   print("num_timepoints is ")
   print(num_timepoints)

   print("flattened_matrix = data.reshape((num_timepoints, num_voxels))")
    
   flattened_matrix = fmri_data.reshape((num_voxels, num_timepoints))
   return flattened_matrix
    


if __name__ == "__main__":
    ###############################################
    #  Step 1: Init - Set up arguments for script
    ###############################################
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

    ###############################################
    #  Step 2: NIfTI Loading and Flattening 
    ###############################################
    print("Load and NIfTI file")
    fmri_img = nib.load(func_filepath)
    
    print("fmri_data = fmri_img.get_fdata()")
    fmri_data = fmri_img.get_fdata()

    num_voxels = np.prod(fmri_data.shape[:-1]) 
    print("num_voxels = np.prod(fmri_data.shape[:-1]) - {}".format(num_voxels))
       
    print("fmri_data.shape : {}".format(fmri_data.shape))

    print("xdim = fmri_data.shape[0]")
    xdim = fmri_data.shape[0]
    ydim = fmri_data.shape[1]
    zdim = fmri_data.shape[2]
    num_vols = fmri_data.shape[3]
    fmri_dimensions = (xdim, ydim, zdim, num_vols)
    print("fmri_dimensions is {}".format(fmri_dimensions))




    ##########################################################################################
    #  Step 3: Perform PCA on subject's flattened matrix using ICA-Torch pca_whiten function
    #########################################################################################
    n_comps = 150 #should this be an argument of the script?

    print("mask_img = nib.load(mask_filepath)")
    mask_img = nib.load(mask_filepath)

    print("mask_data = mask_img.get_fdata()")
    mask_data = mask_img.get_fdata()

    print("boolean_mask = np.asarray(mask_data, dtype=bool)")
    boolean_mask = np.asarray(mask_data, dtype=bool)

    print("brain_voxels = fmri_data[boolean_mask]")
    brain_voxels = fmri_data[boolean_mask]

    print("num_brain_voxels = boolean_mask.sum()")
    num_brain_voxels = boolean_mask.sum()
    
    print("flattened_data = brain_voxels.reshape((num_vols, num_brain_voxels))")
    flattened_data = brain_voxels.reshape((num_vols, num_brain_voxels))
    

    print("flattened_data.shape : {}".format(flattened_data.shape))

    # Run PCA on the flattened data matrix of brain voxels
    print("x_white, white, dewhite = pca_whiten(flattened_data.T, n_comps)")
    x_white, white, dewhite = pca_whiten(flattened_data, n_comps)
    print("x_white.shape: {}".format(x_white.shape))


    # Fill in result of PCA on brain voxels into correct indices
    print("Fill in result of PCA on brain voxels into correct indices")
    pca_res_flat = np.zeros((n_comps, num_voxels))

    print("boolean_mask.shape {}".format(boolean_mask.shape))
    print("pca_res_flat.shape {}".format(pca_res_flat.shape))
    print("x_white.shape: {}".format(x_white.shape))
    print("does x_white contain any 1s {}".format(np.asarray(x_white, dtype=bool).sum()))


    boolean_mask_flat = boolean_mask.reshape((902629))
    print("boolean_mask_flat.shape {}".format(boolean_mask_flat.shape))

    print("how many 1s in boolean_mask_flat {}".format(boolean_mask_flat.sum()))
    
    print("type(boolean_mask_flat) {}".format(type(boolean_mask_flat)))
    
    print("type(boolean_mask_flat[0]) {}".format(type(boolean_mask_flat[0])))
    

    for i in range(x_white.shape[0]):#num comps
        for j in range(x_white.shape[1]):#num voxels
            if boolean_mask_flat[j]:

                #print("inserting value into pca_res_flat {}".format(white_T[i]))
                pca_res_flat[i][j] = x_white[i][j]
        else:
            pca_res_flat[i][j] = 0

    print("pca_res_flat {} ".format(pca_res_flat))

    # func --> fMRI data
    # f --> data has been flattened
    # m --> data has been masked
    # w --> data has been whitened
    # pca --> pca has been applied to the data
    prefix = "func_f_m_w_pca"


    #####################################################
    #  Step 4: (Optional) Save data as NIfTI for visual inspection
    #####################################################
    save_data_as_nifti(pca_res_flat, output_dir, sub_id, prefix, fmri_dimensions, n_comps)
    

    ###########################################################################
    #  Step 5: Save flattened data as binary file to use during group analysis
    ###########################################################################
    save_mat_as_bin(pca_res_flat, output_dir, sub_id, prefix)