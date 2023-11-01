#!/bin/bash

module load matlab

FMRI_NIFTI=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/example.nii
SM_NIFTI=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/pooled_47.nii
SCRIPT=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/niiToMatrix.m

matlab -batch "setenv('inputArg1', '${FMRI_NIFTI}'); setenv('inputArg2', '${SM_NIFTI}'); run('${SCRIPT}')"
