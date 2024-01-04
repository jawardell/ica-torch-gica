#!/bin/bash

# directory of preprocessed fmri data
PREP_DIR=/data/users2/jwardell1/nshor_docker/examples/fbirn-project

# capture subject IDs from raw data directory
SUBS_FILE=${PREP_DIR}/FBIRN/subjects.txt
num_subs=`cat $SUBS_FILE | wc -l`
IFS=$'\n' sub_ids=($(cat $SUBS_FILE))


# create empty paths file to store filepaths to input data
ICAR_DIR=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/python_work
GROUP_DIR=/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis
PATHS_FILE=${ICAR_DIR}/datasets/fbirn-project/paths


# delete the paths file if it already exists
if [ -f $PATHS_FILE ]; then
	rm $PATHS_FILE
else
	touch $PATHS_FILE
fi

TEMPLATE_FILE=${GROUP_DIR}/Neuromark_fMRI_1.0.nii

# write filepaths for all input data to paths file for each subject
for(( i=0; i<$num_subs; i++ ))
do
	subjectID=${sub_ids[$i]} # grab subject id from array at index i
	echo "${subjectID}" >> $PATHS_FILE # first argument of subject_level_pca.py
	echo "${PREP_DIR}/FBIRN/${subjectID}/ses_01/processed/${subjectID}_rest.nii.gz" >>  $PATHS_FILE #second argument of subject_level_pca.py
	echo "${PREP_DIR}/FBIRN/${subjectID}/ses_01/processed" >> $PATHS_FILE #third argument of subject_level_pca.py
	echo "${PREP_DIR}/FBIRN/group_mean_masks/groupmeanmask_3mm.nii" >> $PATHS_FILE #fourth argument of subject_level_pca.py
	echo "${TEMPLATE_FILE}" >> $PATHS_FILE #fifth argument of subject_level_pca.py
done
