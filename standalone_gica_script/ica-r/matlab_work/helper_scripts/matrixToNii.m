addpath("/trdapps/linux-x86_64/matlab/toolboxes/dicm2nii/")

% Set constant dimensions (modify as needed)
xdim = 53;
ydim = 63;
zdim = 46;

% Load ICOutMax data
load('ICOutMax.mat');  % Load the ICOutMax.mat file


% Get header from example.nii
exampleNifti = nii_tool('load', 'example.nii');


% Print the shape of the loaded matrix
disp('Shape of ICOutMax:');
disp(size(ICOutMax));

ICOutMax = ICOutMax';

% Get the number of time points from the loaded data
num_timepoints = size(ICOutMax, 2);
disp(num_timepoints)

% Reshape the data into a 4D tensor
ICOutMax = reshape(ICOutMax, xdim, ydim, zdim, num_timepoints);

% Create a NIfTI structure
nii = nii_tool('init', ICOutMax);

% Set the image data of the new NIfTI structure
nii.img = ICOutMax;  % Assuming ICOutMax contains your image data

% Use the header from 'example.nii' for the new NIfTI structure
nii.hdr = exampleNifti.hdr;

% Save the NIfTI file using nii_tool
nii_tool('save', nii, 'ICOutMax.nii');
