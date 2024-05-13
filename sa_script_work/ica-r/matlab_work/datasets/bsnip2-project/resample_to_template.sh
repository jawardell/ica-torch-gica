#!/bin/bash

set -x

PROJECT_DIR=/data/users2/jwardell1/nshor_docker/examples/bsnip2-project


TEMPLATE=/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii

module load afni

IFS=$'\n' sites=(`ls -1 $PROJECT_DIR/BSNIP2`)
for site in "${sites[@]}"
do
	cd ${PROJECT_DIR}/BSNIP2/$site
	IFS=$'\n' subjects=(`ls -1`)
	for subject in "${subjects[@]}"
	do
		if [ -d ${PROJECT_DIR}/BSNIP2/$site/$subject/ses_01/processed ];then
			cd ${PROJECT_DIR}/BSNIP2/$site/$subject/ses_01/processed
			PROCSTR=rest.nii.gz
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
					echo "procfile $PROCFILE not found for subject $subject site $site"
				elif [ -f func_resampled.nii ];then
					echo "func file already found, skipping"
				else
					3dresample -master $TEMPLATE -prefix func_resampled.nii -input $PROCFILE
				fi
			done
		else
			echo "session not processed for subject $subject site $site"
		fi
	done
done
