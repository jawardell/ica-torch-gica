addpath("/trdapps/linux-x86_64/matlab/toolboxes/")

try
    nifti;
    disp('NIfTI toolbox is installed.');
catch
    disp('NIfTI toolbox is not installed.');
end
