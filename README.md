# Group ICA Script in Python for Preprocessed Data

This is part of the fMRI Preprocessing for Causal Modeling Project. 
The goal of this is to perform the same task as 
<a href="https://github.com/trendscenter/aa-gift">aa-gift</a> but without using GIFT 
and using <a href="https://github.com/bbradt/ica-torch">ICA-Torch</a> instead.

## Container Files
The `container_data` directory contains work toward using GIFT software to run ICA in a singularity container. 

These files might be adapted to later accomdate the standalone group ica python script. 


## Standalone Group ICA Script
The `standalone_group_ica_script` directory contains a script that is intended to run group ICA on a study's set of preprocessed fMRI files. 

The directory also contains a subdirectory called `input_data` which contains a script for generating an input string 
of an entire study's processed fMRI filepaths. 

Each filepath is the full path to each subject's processed fMRI data with each subject's fMRI file separated by a string.

### The `group_ica_script.py` Python Script

This script operates on a study's fMRI data and produces _N_ spatial maps, where _N_ is the number of subjects. 

Each spatial map is saved as `${subjectID}_SM.nii.gz` where `${subjectID}` should be replaced by the subject's unique identifier. 
Each spatial map file contains _K_ volumes where _K_ is the number of components estimated by ICA. 
Each volume in the spatial map file represents an activation network that was detected via ICA. 







