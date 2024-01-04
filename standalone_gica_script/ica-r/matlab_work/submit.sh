#!/bin/bash

project_dir=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work
paths_file=${project_dir}/paths

num_lines=`wc -l <  $paths_file`
num_total_runs=$(( $num_lines / 5 ))

runix=$(( $num_total_runs - 1 ))

sbatch --array=0-${runix} ${project_dir}/gigicar.job
#sbatch --array=0-${runix}%10 ${project_dir}/gigicar.job
