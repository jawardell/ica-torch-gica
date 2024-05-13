#!/bin/bash


if [ -z $1 ]; then
	export SLURM_ARRAY_TASK_ID=0
else
	export SLURM_ARRAY_TASK_ID=$1
fi

export PATHS_FILE=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/datasets/ds001747-project/paths_ds001747


module load matlab

IFS=$'\n'
export paths_array=($(cat ${PATHS_FILE}))

export func_ix=$(( 5*$SLURM_ARRAY_TASK_ID ))
export sm_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 1 ))
export mask_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 2 ))
export sub_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 3 ))
export out_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 4 ))


export FMRI_NIFTI=${paths_array[${func_ix}]}
echo "FMRI_NIFTI: $FMRI_NIFTI"

export SM_NIFTI=${paths_array[${sm_ix}]}
echo "SM_NIFTI: $SM_NIFTI"


export MASK_NIFTI=${paths_array[${mask_ix}]}
echo "MASK_NIFTI: $MASK_NIFTI"


export SUBID=${paths_array[${sub_ix}]}
echo "SUBID: $SUBID"


export OUTPUT_DIR=${paths_array[${out_ix}]}
echo "OUTPUT_DIR: $OUTPUT_DIR"


export SCRIPT=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/gigicar.m


#matlab -batch "setenv('inputArg1', '${FMRI_NIFTI}'); setenv('inputArg2', '${SM_NIFTI}'); setenv('inputArg3', '${SUBID}'); setenv('inputArg4', '${MASK_NIFTI}'); setenv('inputArg5', '${OUTPUT_DIR}'); run('${SCRIPT}')"

