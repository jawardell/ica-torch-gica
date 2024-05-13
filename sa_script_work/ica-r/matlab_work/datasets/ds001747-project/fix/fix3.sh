#!/bin/bash
set -x

# Define the base directory
base_dir="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/"

# Loop through each subject ID in subjects.txt
while IFS= read -r sub_id; do
    # Unzip TR100 files
    gunzip "${base_dir}${sub_id}/processed/${sub_id}_mreg_data_TR100_resampled.nii.gz"

    # Unzip TR2150 files
#    gunzip "${base_dir}${sub_id}/processed/${sub_id}_data_TR2150_resampled.nii.gz"

done < subjects.txt

