#!/bin/bash

project_dir=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script
paths_file=${project_dir}/input_data/proc_fmri_datafiles.txt

SUBS_FILE=subjects.txt
num_subs=`cat $SUBS_FILE | wc -l`
num_total_runs=$(( $num_subs / 5 ))

runix=$(( $num_total_runs - 1 ))


#batch_size=10
#sbatch --array=0-${runix}%${batch_size} ${project_dir}/procruns.job

sbatch --array=0-${runix} ${project_dir}/ica-r/procruns.job
