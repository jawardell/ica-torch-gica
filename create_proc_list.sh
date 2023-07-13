#!/bin/bash

subs_file=/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/subjects.txt
num_subs=$(wc -l < "$subs_file")

IFS=$'\n' sub_ids=($(cat ${subs_file}))

touch proc_fmri_datafiles.txt

for(( i=0; i<$num_subs; i++))
do
	subjectID=${sub_ids[$i]}
	echo -n "/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/${subjectID}/ses_01/processed/rest_processed.nii.gz," >> proc_fmri_datafiles.txt
done
