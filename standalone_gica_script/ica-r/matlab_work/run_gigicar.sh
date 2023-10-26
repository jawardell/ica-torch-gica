#!/bin/bash

module load matlab

FMRI_MATRIX=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/subject_data.mat
SM_MATRIX=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/group_data.mat
SCRIPT=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/run_gigicar.m

matlab -batch "setenv('inputArg1', '${FMRI_MATRIX}'); setenv('inputArg2', '${SM_MATRIX}'); run('${SCRIPT}')"
