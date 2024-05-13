#!/bin/bash

project_dir=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work
paths_file=${project_dir}/datasets/ds001747-project/paths_ds001747

num_lines=`wc -l <  $paths_file`
num_total_runs=$(( $num_lines / 5 ))

startix=0
endix=$(( $num_total_runs - 1 ))
#endix=9

sbatch --array=${startix}-${endix} ${project_dir}/datasets/ds001747-project/gigicar.job
#sbatch --array=0-${runix}%10 ${project_dir}/gigicar.job
