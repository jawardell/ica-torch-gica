#!/bin/bash

set -x

PROJECT_DIR=/data/users2/jwardell1/nshor_docker/examples/fbirn-project

SUBJECTS_FILE=${PROJECT_DIR}/FBIRN/subjects.txt
IFS=$'\n' subjects=(`cat $SUBJECTS_FILE`)

if [ -z $1 ];then
	TEMPLATE=/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii
else
	TEMPLATE=$1
fi


module load afni

for subject in "${subjects[@]}"
do
	cd ${PROJECT_DIR}/FBIRN/$subject
	IFS=$'\n' sessions=(`ls -1`)
	for session in "${sessions[@]}"
	do
		if [ -d ${PROJECT_DIR}/FBIRN/$subject/$session/processed ];then
			cd ${PROJECT_DIR}/FBIRN/$subject/$session/processed
			PROCSTR=${session}_rest_open_evenURSIfirst_
			PROCFILE=''
			IFS=$'\n' files=(`ls -1`)
			for file in ${files[@]}
			do
				echo "${files[@]}"
				if [[ "$file" == *"${PROCSTR}"* ]];then
					PROCFILE=$file
					echo $PROCFILE
				fi
				if [ -z $PROCFILE ]; then
					echo "procfile $PROCFILE not found for subject $subject session $session"
				elif [ -f func_resampled.nii ];then
					echo "func file already found, skipping"
				else
					3dresample -master $TEMPLATE -prefix func_resampled.nii -input $PROCFILE
				fi
			done
		else
			echo "session $session not processed for subject $subject"
		fi
	done
done
