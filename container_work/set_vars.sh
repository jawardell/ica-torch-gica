#!/bin/bash
export GSU_ID="jwardell1"
export dataset="fbirn"
export dataset_cp=`capitalize ${dataset}`
export subjects_file="/data/users2/${GSU_ID}/nshor_docker/examples/${dataset}-project/${dataset_cp}/subjects.txt"
export my_root=`pwd`
export output_file="${my_root}/proc_fmri_datafiles.txt"
export INFILES=`cat $output_file`
#export INFILES="/input/000300655084/ses_01/processed/rest_processed.nii.gz"
export OUTFILE="/data/users2/${GSU_ID}/nshor_docker/examples/${dataset}-project/GICA"
export ALGORITHM="gica"
export SIFFILE="fmri-gica-sc.sif"
export SCRIPTNAME="run_gift.py"
export DATA=/data/users2/${GSU_ID}/nshor_docker/examples/${dataset}-project/${dataset_cp}
