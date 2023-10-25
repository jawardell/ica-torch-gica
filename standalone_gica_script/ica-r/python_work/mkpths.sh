#!/bin/bash

# directory of raw fmri data to calculate number of subjects
DATA_DIR=/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data

# directory of preprocessed fmri data
PROJ_DIR=/data/users2/jwardell1/nshor_docker/examples/fbirn-project

# capture subject IDs from raw data directory
SUBS_FILE=${PROJ_DIR}/FBIRN/subjects.txt
num_subs=`cat $SUBS_FILE | wc -l`
IFS=$'\n' sub_ids=($(cat $SUBS_FILE))


# create empty paths file to store filepaths to input data
GICA_DIR=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script
BRECON_DIR=${GICA_DIR}/ica-r
PATHS_FILE=${BRECON_DIR}/paths

if [ -f $PATHS_FILE ]; then
	rm $PATHS_FILE
else
	touch $PATHS_FILE
fi

TEMPLATE_FILE=${GICA_DIR}/group_level_analysis/NeuroMark_resampled.nii

# write filepaths for all input data to paths file for each subject
for(( i=0; i<$num_subs; i++ ))
do
	subjectID=${sub_ids[$i]} # grab subject id from array at index i
	echo "${subjectID}" >> $PATHS_FILE # first argument of subject_level_pca.py
	echo "${PROJ_DIR}/FBIRN/${subjectID}/ses_01/processed/${subjectID}_rest.nii.gz" >>  $PATHS_FILE #second argument of subject_level_pca.py
	echo "${PROJ_DIR}/FBIRN/${subjectID}/ses_01/processed" >> $PATHS_FILE #third argument of subject_level_pca.py
	echo "${PROJ_DIR}/FBIRN/group_mean_masks/groupmeanmask_3mm.nii" >> $PATHS_FILE #fourth argument of subject_level_pca.py
	echo "${TEMPLATE_FILE}" >> $PATHS_FILE #fifth argument of subject_level_pca.py
done
