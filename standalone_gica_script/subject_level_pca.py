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


def save_data_as_nifti(X, outpath, sub_id, prefix):
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
    xdim = X.shape[0]
    ydim = X.shape[1]
    zdim = X.shape[2]
    num_vols = X.shape[3]


    # Create an empty 4D array to store all the 3D images
    print("image_stack = np.zeros((xdim, ydim, zdim, num_vols))")
    image_stack = np.zeros((xdim, ydim, zdim, num_vols))

    print("image_stack.shape  {}".format(image_stack.shape))
    print("X.shape {}".format(X.shape))
   

    # Create a NIfTI image object from the stacked 4D array
    print("nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))")
    nifti_img = nib.Nifti1Image(X, affine=np.eye(4))

    # Save the NIfTI image to a file
    nifti_file = "{}/{}_{}.nii.gz".format(outpath, sub_id, prefix)
    print("nifti_file = {}".format(nifti_file))
    
    nib.save(nifti_img, nifti_file)



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



    # load your data
    print("fmri_img = nib.load(func_filepath)")
    fmri_img = nib.load(func_filepath) # this is currently the pca result, but it will be all your subject's timepoints otherwise, the goal is to get to `data` tensor (below)

    print("mask = nib.load(mask_filepath)")
    mask = nib.load(mask_filepath) # the mask

    print("idx = np.where(mask.dataobj)")
    idx = np.where(mask.dataobj) # indices of non-zero elements
    print("idx {}".format(idx))

    print("data = fmri_img.get_fdata()")
    data = fmri_img.get_fdata() # this is the data of a subject with the last dimension being the total time duration

    print("data2pca = data[*idx,:]")
    data2pca = data[*idx,:] # this will be voxel by time data you need to apply the PCA to

    # ... apply your PCA here to the appropriately transposed data to obtained pca_result
    print("pca_result = pca_whiten(data2pca.T, n_comps)")
    pca_result, white, dewhite = pca_whiten(data2pca.T, n_comps)
    
    print("pca_result.shape  {}".format(pca_result.shape))

    print("pca_volumes = np.zeros(data.shape[:3], n_comps)")
    pca_volumes = np.zeros((data.shape[0], data.shape[1], data.shape[2],  n_comps))
    
    print("pca_volumes[*idx,:] = pca_result")
    #pca_volumes[*idx,:] = pca_result
    pca_volumes[*idx,:] = pca_result.T


    print("pca_volumes.shape {}".format(pca_volumes.shape))
    
    #####################################################
    #  Step 4: (Optional) Save data as NIfTI for visual inspection
    #####################################################
    prefix = "fmri_pca_w"
    save_data_as_nifti(pca_volumes, output_dir, sub_id, prefix)
    

    ###########################################################################
    #  Step 5: Save flattened data as binary file to use during group analysis
    ###########################################################################
    save_mat_as_bin(pca_volumes, output_dir, sub_id, prefix)