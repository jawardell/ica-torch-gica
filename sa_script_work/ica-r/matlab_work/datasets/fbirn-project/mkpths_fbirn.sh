#!/bin/bash

DATA_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN
PROJECT_DIRECTORY=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/matlab_work/datasets/fbirn-project


sub_file=${PROJECT_DIRECTORY}/subjects.txt
num_subs=$(cat $sub_file | wc -l)
IFS=$'\n' sub_ids=($(cat $sub_file))

PATH_FILE=${PROJECT_DIRECTORY}/paths_fbirn

touch ${PATH_FILE}

for(( i=0; i<$num_subs; i++))
do
	subjectID=${sub_ids[$i]}
	#FMRI_NIFTI
	echo "${DATA_DIRECTORY}/${subjectID}/ses_01/processed/func_resampled.nii" >> $PATH_FILE
	#SM_NIFTI
	echo "/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii" >> $PATH_FILE
	#MASK_NIFTI
	echo "${DATA_DIRECTORY}/group_mean_masks/mask_resampled.nii" >> $PATH_FILE
	#SUBID
	echo "${subjectID}" >> $PATH_FILE
	#OUTPUT_DIRECTORY
	OUTPUT_DIRECTORY=${DATA_DIRECTORY}/${subjectID}/ses_01/processed
	echo "${OUTPUT_DIRECTORY}" >> $PATH_FILE
done
