#!/bin/bash

input_file="/data/users2/jwardell1/aa-gift/proc_fmri_datafiles.bak"
replace_path="/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/\${subject_id}/ses_01"
subjects_file="/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/subjects.txt"

while IFS= read -r subject_id; do
  sed -i "s|/input/|${replace_path}|g" "$input_file"
done < "$subjects_file"
