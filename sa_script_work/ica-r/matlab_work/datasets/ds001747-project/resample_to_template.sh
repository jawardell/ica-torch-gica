#!/bin/bash

set -x
set -e

PROJECT_DIR=/data/users2/jwardell1/nshor_docker/examples/ds001747-project

SUBJECTS_FILE=${PROJECT_DIR}/ds001747/subjects.txt
IFS=$'\n' subjects=(`cat $SUBJECTS_FILE`)

TEMPLATE=/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii


module load afni

for subject in "${subjects[@]}"
do
	cd ${PROJECT_DIR}/ds001747/${subject}/processed
	PROCFILE=${subject}_${subject}_task-rest_run-01_bold.nii
	if [ -f $PROCFILE ]; then
		if [ -f func_resampled.nii ];then
			echo "func_resampled.nii alredy found, skipping."
		else
			3dresample -master $TEMPLATE -prefix func_resampled.nii -input $PROCFILE
		fi
	else
		echo "procfile $PROCFILE not found for subject $subject session $session"
	fi
done
