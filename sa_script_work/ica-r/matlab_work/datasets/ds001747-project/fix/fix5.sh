#!/bin/bash

# Loop through each subject directory
while IFS= read -r sub_id; do
    # Create the 'bak' directory if it doesn't exist
    mkdir -p "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak"

    # Move the specified files to the 'bak' directory
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}_TR100.mat" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak/"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/TCOutMax_${sub_id}_TR100.mat" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak/"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}_TR2150.mat" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak/"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/TCOutMax_${sub_id}_TR2150.mat" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak/"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}_TR2150.nii" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak/"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/TCOutMax_${sub_id}_TR2150.mat."* "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak/"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}_SANITYCHECK.nii" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/bak/"
done < subjects.txt

