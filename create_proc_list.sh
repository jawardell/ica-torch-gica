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
