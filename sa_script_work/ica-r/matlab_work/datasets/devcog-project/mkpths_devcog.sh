#!/bin/bash

DATA_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/devcog-project/DEVCOG
PROJECT_DIRECTORY=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/datasets/devcog-project


sub_file=${PROJECT_DIRECTORY}/subjects.txt

PATH_FILE=${PROJECT_DIRECTORY}/paths_devcog

> ${PATH_FILE}

IFS=$'\n' subjects=(`cat $sub_file`)

for subject in "${subjects[@]}"
do
	cd ${DATA_DIRECTORY}/$subject
	IFS=$'\n' sessions=(`ls -1`)
	for session in "${sessions[@]}"
	do
		cd ${DATA_DIRECTORY}/$subject/$session
		if [ -d ${DATA_DIRECTORY}/$subject/$session/processed ];then 
		
			#FMRI_NIFTI
			echo "${DATA_DIRECTORY}/${subject}/$session/processed/func_resampled.nii" >> $PATH_FILE
			#SM_NIFTI
			echo "/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii" >> $PATH_FILE
			#MASK_NIFTI
			echo "${DATA_DIRECTORY}/group_mean_masks/mask_resampled.nii" >> $PATH_FILE
			#SUBID
			echo "${subject}" >> $PATH_FILE
			#OUTPUT_DIRECTORY
			OUTPUT_DIRECTORY=${DATA_DIRECTORY}/${subject}/$session/processed
			echo "${OUTPUT_DIRECTORY}" >> $PATH_FILE
		else
			echo "session $session not processed for subject $subject"
		fi
	done
done
