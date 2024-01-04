#!/bin/bash

DATA_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP
PROJECT_DIRECTORY=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work


sub_file=${PROJECT_DIRECTORY}/subjects_hcp.txt
num_subs=$(cat $sub_file | wc -l)
IFS=$'\n' sub_ids=($(cat $sub_file))

PATH_FILE=${PROJECT_DIRECTORY}/paths_hcp

touch ${PATH_FILE}

for(( i=0; i<$num_subs; i++))
do
	subjectID=${sub_ids[$i]}
	#FMRI_NIFTI
	#377451_377451_3T_rfMRI_REST1_RL.nii.gz
	echo "${DATA_DIRECTORY}/${subjectID}/processed/${subjectID}_${subjectID}_3T_rfMRI_REST1_RL.nii" >> $PATH_FILE
	#SM_NIFTI
	echo "/data/users2/jwardell1/ica-torch-gica/group_level_analysis/NeuroMark_resampled.nii" >> $PATH_FILE
	#MASK_NIFTI
	echo "${DATA_DIRECTORY}/group_mean_masks/groupmeanmask_3mm.nii" >> $PATH_FILE
	#SUBID
	echo "${subjectID}" >> $PATH_FILE
	#OUTPUT_DIRECTORY
	OUTPUT_DIRECTORY=${DATA_DIRECTORY}/${subjectID}/processed
	echo "${OUTPUT_DIRECTORY}" >> $PATH_FILE
done
