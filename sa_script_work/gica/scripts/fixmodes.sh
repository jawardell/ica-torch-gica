#!/bin/bash

sla_filepaths=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/input_data/sub_level_pca_files.txt


IFS=$',' file_paths=($(cat $sla_filepaths))

num_subs=368

for(( i=0; i<$num_subs; i++))
do
	sla_file=${file_paths[$i]}
	#chmod 777 $sla_file
	rm $sla_file
	
done
