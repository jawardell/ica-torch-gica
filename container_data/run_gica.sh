#!/bin/bash

module load singularity


GSU_ID="jwardell1"

dataset="fbirn"
dataset_cp=`capitalize ${dataset}`

subjects_file="/data/users2/${GSU_ID}/nshor_docker/examples/${dataset}-project/${dataset_cp}/subjects.txt"
my_root=`pwd`
output_file="${my_root}/proc_fmri_datafiles.txt"

if [ -f $output_file ]; then 
	rm $output_file
else
	touch $output_file
fi

# Loop through each subject directory
while IFS= read -r subject_id; do
  # Check if the subject directory exists
  subject_dir="/data/users2/${GSU_ID}/nshor_docker/examples/${dataset}-project/${dataset_cp}/${subject_id}/ses_01/processed"
  if [ -d "$subject_dir" ]; then
    # Find the .nii.gz file within the subject directory
    file_path=`ls $subject_dir | grep *.nii.gz`
    
    # Echo the file path to the output file
    filename="rest_processed.nii.gz"
    echo -n "/input/${subject_id}/ses_01/processed/$filename," >> "$output_file"
  fi
done < "$subjects_file"



#INFILES=`cat $output_file`
INFILES="/input/000300655084/ses_01/processed/rest_processed.nii.gz"
OUTFILE="/data/users2/${GSU_ID}/nshor_docker/examples/${dataset}-project/GICA"
ALGORITHM="gica"

SIFFILE="fmri-gica-sc.sif"
SCRIPTNAME="run_gift.py"
DATA=/data/users2/${GSU_ID}/nshor_docker/examples/${dataset}-project/${dataset_cp}

#/input/000300655084/ses_01/processed/rest_processed.nii.gz

singularity exec --bind ${OUTFILE}:/out,${DATA}:/input ${SIFFILE} /app/${SCRIPTNAME} -a ${ALGORITHM} -i ${INFILES} -o /out 
