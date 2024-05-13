#!/bin/bash

# Loop through each subject directory
while IFS= read -r sub_id; do
    # Rename the files in the subject directory
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}.mat" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}_TR100.mat"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}.nii" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/ICOutMax_${sub_id}_TR100.nii"
    mv "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/TCOutMax_${sub_id}.mat" "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${sub_id}/processed/TCOutMax_${sub_id}_TR100.mat"
done < subjects.txt

