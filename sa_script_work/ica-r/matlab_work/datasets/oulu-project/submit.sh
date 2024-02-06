#!/bin/bash

project_dir=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work
paths_file=${project_dir}/datasets/oulu-project/paths_oulu

num_lines=`wc -l <  $paths_file`
num_total_runs=$(( $num_lines / 5 ))

startix=0
#endix=$(( $num_total_runs - 1 ))
endix=10

sbatch --array=${startix}-${endix} ${project_dir}/datasets/oulu-project/gigicar.job
#sbatch --array=0-${runix}%10 ${project_dir}/gigicar.job
