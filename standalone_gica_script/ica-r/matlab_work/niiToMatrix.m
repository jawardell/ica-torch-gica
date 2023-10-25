addpath("/trdapps/linux-x86_64/matlab/toolboxes/dicm2nii/")

try
	nii_tool_version = nii_tool('default');
	disp("The dicm2nii toolbox was loaded correctly.")
catch exception
	disp("The dicm2nii toolbox was not loaded correctly.")
end
	

% Parse arguments
subjectNiftiFile = getenv("inputArg1");
groupNiftiFile = getenv("inputArg2");

% Display the arguments
disp(["Subject Nifti File: " subjectNiftiFile]);
disp(["Group Nifti File: " groupNiftiFile]);


% Load the NIfTI files
subjectNifti = nii_tool('load', subjectNiftiFile); % Load subject fMRI data
groupNifti = nii_tool('load', groupNiftiFile);     % Load group maps data

% Get the dimensions of the NIfTI data
subjectDims = size(subjectNifti.img);
groupDims = size(groupNifti.img);

% Reshape the NIfTI data into 2D arrays
subjectData = reshape(subjectNifti.img, [], subjectDims(4));  % (voxels, timepoints)
groupData = reshape(groupNifti.img, [], groupDims(4));       % (voxels, group spatial maps)

% Transpose the data if needed to match your desired dimensions
subjectData = subjectData';  % (timepoints, voxels)
groupData = groupData';      % (group spatial maps, voxels)

% Print the shape of the matrices
disp('Shape of subjectData:');
disp(size(subjectData));

disp('Shape of groupData:');
disp(size(groupData));


% Save the matrices as .mat files in the current working directory
save("subject_data.mat", "subjectData");
save("group_data.mat", "groupData");
