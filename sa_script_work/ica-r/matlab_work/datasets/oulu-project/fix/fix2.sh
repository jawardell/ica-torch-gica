#!/bin/bash

# Enable strict mode
set -e

# Define the path to the main directory
main_dir="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU"

# Loop through each subject ID in subjects.txt
while IFS= read -r sub_id; do
    # Define the path to the processed directory
    processed_dir="${main_dir}/${sub_id}/processed"

    # Define the filenames
    mat_file="TCOutMax_${sub_id}_TR2150.mat"
    bak_file="TCOutMax_${sub_id}_TR2150.mat.bak"

    # Perform the renaming if .bak file exists
    if [[ -f "${processed_dir}/${bak_file}" ]]; then
        mv "${processed_dir}/${mat_file}" "${processed_dir}/${mat_file}.$(date +%Y%m%d%H%M%S)"
        mv "${processed_dir}/${bak_file}" "${processed_dir}/${mat_file}"
    fi
done < subjects.txt

