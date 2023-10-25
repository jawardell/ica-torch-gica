#!/bin/bash

SLURM_ARRAY_TASK_ID=0

export PATH=/data/users2/jwardell1/miniconda3/bin:$PATH

source /data/users2/jwardell1/miniconda3/etc/profile.d/conda.sh

conda activate /data/users2/jwardell1/miniconda3/envs/ica-torch


GICA_DIR=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script

PATHS_FILE=${GICA_DIR}/ica-r/paths

IFS=$'\n'
paths_array=($(cat ${PATHS_FILE}))
sub_ix=$(( 5*$SLURM_ARRAY_TASK_ID ))
func_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 1 ))
out_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 2 ))
mask_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 3 ))
template_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 4 ))

sub_id=${paths_array[${sub_ix}]}
func_file=${paths_array[${func_ix}]}
out_dir=${paths_array[${out_ix}]}
mask_file=${paths_array[${mask_ix}]}
template_file=${paths_array[${template_ix}]}


###degbugging purposes only
func_file=example.nii
template_file=pooled_47.nii
