#!/bin/bash
set -e

# Define the path to the main directory
main_dir="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU"

# Loop through each subject ID in subjects.txt
while IFS= read -r sub_id; do
    # Define the path to the subject directory
    sub_dir="${main_dir}/${sub_id}"

    # Define the paths to the processed and bak directories
    processed_dir="${sub_dir}/processed"
    bak_dir="${processed_dir}/bak"
    if [ -d $bak_dir ]; then

	    # Move data out of the bak directory to the processed directory
	    mv "${bak_dir}"/* "${processed_dir}/"

	    # Remove the bak directory
	    rmdir "${bak_dir}"
     fi
done < subjects.txt

