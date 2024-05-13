#!/bin/bash

set -x

DATA_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/bsnip2-project/BSNIP2
PROJECT_DIRECTORY=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/datasets/bsnip2-project


PATH_FILE=${PROJECT_DIRECTORY}/paths_bsnip2

> ${PATH_FILE}

IFS=$'\n' sites=(`ls -1 ${DATA_DIRECTORY}`)

for site in "${sites[@]}"
do
	cd ${DATA_DIRECTORY}/$site
	IFS=$'\n' subjects=(`ls -1`)
	for subject in "${subjects[@]}"
	do
		cd ${DATA_DIRECTORY}/$site/$subject
		if [ -d ${DATA_DIRECTORY}/$site/$subject/ses_01/processed ];then 
		
			#FMRI_NIFTI
			echo "${DATA_DIRECTORY}/${site}/$subject/ses_01/processed/func_resampled.nii" >> $PATH_FILE
			#SM_NIFTI
			echo "/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii" >> $PATH_FILE
			#MASK_NIFTI
			echo "${DATA_DIRECTORY}/group_mean_masks/mask_resampled.nii" >> $PATH_FILE
			#SUBID
			echo "${subject}" >> $PATH_FILE
			#OUTPUT_DIRECTORY
			OUTPUT_DIRECTORY=${DATA_DIRECTORY}/${site}/$subject/ses_01/processed
			echo "${OUTPUT_DIRECTORY}" >> $PATH_FILE
		else
			echo "subject $subject not processed for site $site"
		fi
	done
done
