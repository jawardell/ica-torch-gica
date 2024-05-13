#!/bin/bash

# Define the path to the processed folder
processed_folder="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/*/processed"

# Loop through each subject ID in subjects.txt
while IFS= read -r sub_id; do
    # Define the filenames
    ic_input="ICOutMax_${sub_id}.nii"
    ic_output="ICOutMax_${sub_id}_TR2150.nii"
    max_input="ICOutMax_${sub_id}.mat"
    max_output="ICOutMax_${sub_id}_TR2150.mat"
    tc_input="TCOutMax_${sub_id}.mat"
    tc_output="TCOutMax_${sub_id}_TR2150.mat"

    # Loop through each processed folder
    for folder in $processed_folder; do
        # Move IC files
        if [ -f "${folder}/${ic_input}" ]; then
            mv "${folder}/${ic_input}" "${folder}/${ic_output}"
        fi

        # Move Max files
        if [ -f "${folder}/${max_input}" ]; then
            mv "${folder}/${max_input}" "${folder}/${max_output}"
        fi

        # Move TC files
        if [ -f "${folder}/${tc_input}" ]; then
            mv "${folder}/${tc_input}" "${folder}/${tc_output}"
        fi
    done
done < subjects.txt

