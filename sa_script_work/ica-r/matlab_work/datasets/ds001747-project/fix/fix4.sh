#!/bin/bash

# Define the base directory
base_dir="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/"

# Loop through each subject ID in subjects.txt
while IFS= read -r sub_id; do
    # Define the source and destination paths
    source_file="${base_dir}${sub_id}/processed/${sub_id}_${sub_id}_data_TR2150_resampled.nii"
    destination_file="${base_dir}${sub_id}/processed/${sub_id}_data_TR2150_resampled.nii"

    # Move the file
    mv "$source_file" "$destination_file"

done < subjects.txt

