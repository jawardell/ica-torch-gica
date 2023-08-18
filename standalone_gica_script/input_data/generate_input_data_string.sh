#!/bin/bash

subs_file=/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/subjects.txt
num_subs=$(wc -l < "$subs_file")

IFS=$'\n' sub_ids=($(cat ${subs_file}))

for(( i=0; i<$num_subs; i++))
do
	subjectID=${sub_ids[$i]}
	echo -n "/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/${subjectID}/ses_01/processed/${subjectID}_pca.torch," >> sub_level_pca_files.txt
done
