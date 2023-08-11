#!/bin/bash

# Computes num elements needed for SLURM parallel array flag and submits the job

# where the group ica scripts are located
gica_dir=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script

# where the subject level analysis paths file is located
# holds 3-tuple arguments to subject_level_pca.py for all subs
paths_file=${gica_dir}/subject_level_analysis/paths

# this is how many arguments the subject_level_pca.py script takes
num_arguments=4

# calculate SLURM task array size
num_lines=`wc -l <  $paths_file`
num_total_runs=$(( $num_lines / $num_arguments ))

# calculate last position in SLURM parallel array
runix=$(( $num_total_runs - 1))

# process all subjects in parallel via sbatch
#sbatch --array=0-$num_total_runs ${gica_dir}/subject_level_analysis/sub_level_pca.job

# uncomment if wanting to process in batches due to I/O issues
batch_size=15
sbatch --array=0-$runix%${batch_size} ${gica_dir}/subject_level_analysis/sub_level_pca.job
