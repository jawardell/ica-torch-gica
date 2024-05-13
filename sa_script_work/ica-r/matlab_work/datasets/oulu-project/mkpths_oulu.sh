#!/bin/bash

DATA_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU
PROJECT_DIRECTORY=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/datasets/oulu-project


sub_file=${PROJECT_DIRECTORY}/subjects.txt
num_subs=$(cat $sub_file | wc -l)
IFS=$'\n' sub_ids=($(cat $sub_file))

PATH_FILE=${PROJECT_DIRECTORY}/paths_oulu

touch ${PATH_FILE}

for tr in 2150 100
do
	for(( i=0; i<$num_subs; i++))
	do
		subjectID=${sub_ids[$i]}
		#FMRI_NIFTI
		if [ "$tr" -eq "2150" ];then
			echo "${DATA_DIRECTORY}/${subjectID}/processed/${subjectID}_${subjectID}_data_TR2150.nii.gz" >> $PATH_FILE
		else
			echo "${DATA_DIRECTORY}/${subjectID}/processed/${subjectID}_mreg_data_TR100.nii.gz" >> $PATH_FILE
		fi

		#SM_NIFTI
		echo "${DATA_DIRECTORY}/group_mean_masks/Neuromark_fMRI_1.0.nii" >> $PATH_FILE

		#MASK_NIFTI
		if [ "$tr" -eq "2150" ];then
			echo "${DATA_DIRECTORY}/group_mean_masks/mask_resampled_TR2150.nii" >> $PATH_FILE
		else
			echo "${DATA_DIRECTORY}/group_mean_masks/mask_resampled_TR100.nii" >> $PATH_FILE
		fi

		#SUBID
		echo "${subjectID}" >> $PATH_FILE

		#OUTPUT_DIRECTORY
		OUTPUT_DIRECTORY=${DATA_DIRECTORY}/${subjectID}/processed
		echo "${OUTPUT_DIRECTORY}" >> $PATH_FILE
	done
done
