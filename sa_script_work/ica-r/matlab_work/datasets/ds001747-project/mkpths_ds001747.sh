#!/bin/bash

DATA_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/ds001747-project/ds001747
PROJECT_DIRECTORY=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/datasets/ds001747-project


sub_file=${PROJECT_DIRECTORY}/subjects.txt
num_subs=$(cat $sub_file | wc -l)
IFS=$'\n' sub_ids=($(cat $sub_file))

PATH_FILE=${PROJECT_DIRECTORY}/paths_ds001747

> ${PATH_FILE}

for(( i=0; i<$num_subs; i++))
do
	subjectID=${sub_ids[$i]}
	#FMRI_NIFTI
	#sub-2975_sub-2975_task-rest_run-01_bold.nii
	#echo "${DATA_DIRECTORY}/${subjectID}/processed/${subjectID}_${subjectID}_task-rest_run-01_bold.nii" >> $PATH_FILE
	echo "${DATA_DIRECTORY}/${subjectID}/processed/func_resampled.nii" >> $PATH_FILE



	#Group NIFTI
	echo "/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii" >> $PATH_FILE


	#MASK_NIFTI
	echo "${DATA_DIRECTORY}/group_mean_masks/groupmeanmask_resampled.nii" >> $PATH_FILE

	#SUBID
	echo "${subjectID}" >> $PATH_FILE
	#OUTPUT_DIRECTORY
	OUTPUT_DIRECTORY=${DATA_DIRECTORY}/${subjectID}/processed
	echo "${OUTPUT_DIRECTORY}" >> $PATH_FILE
done
