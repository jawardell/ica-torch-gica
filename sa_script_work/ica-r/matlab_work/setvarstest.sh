#!/bin/bash

module load matlab


export SLURM_ARRAY_TASK_ID=0

export PATHS_FILE=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/datasets/fbirn-project/paths_fbirn



export IFS=$'\n'
export paths_array=($(cat ${PATHS_FILE}))
export func_ix=$(( 5*$SLURM_ARRAY_TASK_ID ))
export sm_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 1 ))
export mask_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 2 ))
export sub_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 3 ))
export out_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 4 ))


export FMRI_NIFTI=${paths_array[${func_ix}]}
export SM_NIFTI=${paths_array[${sm_ix}]}
export MASK_NIFTI=${paths_array[${mask_ix}]}
export SUBID=${paths_array[${sub_ix}]}
export OUTPUT_DIR=${paths_array[${out_ix}]}


export SCRIPT=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/gigicar.m

#gunzip ${FMRI_NIFTI}.gz
#matlab -batch "setenv('inputArg1', '${FMRI_NIFTI}'); setenv('inputArg2', '${SM_NIFTI}'); setenv('inputArg3', '${SUBID}'); setenv('inputArg4', '${MASK_NIFTI}'); setenv('inputArg5', '${OUTPUT_DIR}'); run('${SCRIPT}')"
#gzip ${FMRI_NIFTI}
