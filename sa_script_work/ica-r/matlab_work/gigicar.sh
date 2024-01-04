#!/bin/bash

module load matlab

#FMRI_NIFTI=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/example.nii
FMRI_NIFTI=/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/000300655084/ses_01/processed/000300655084_rest.nii
OUTPUT_DIR=/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/000300655084/ses_01/processed


#SM_NIFTI=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/pooled_47.nii
SM_NIFTI=/data/users2/jwardell1/ica-torch-gica/group_level_analysis/NeuroMark_resampled.nii


#MASK_NIFTI=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/group_level_analysis/mask_resampled.nii
MASK_NIFTI=/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/group_mean_masks/groupmeanmask_3mm.nii

SUBID=000300655084

SCRIPT=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/gigicar.m

gunzip ${FMRI_NIFTI}.gz
matlab -batch "setenv('inputArg1', '${FMRI_NIFTI}'); setenv('inputArg2', '${SM_NIFTI}'); setenv('inputArg3', '${SUBID}'); setenv('inputArg4', '${MASK_NIFTI}'); setenv('inputArg5', '${OUTPUT_DIR}'); run('${SCRIPT}')"
gzip ${FMRI_NIFTI}
