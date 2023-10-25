addpath("/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work")
% Load the input matrices from files
FmriMatr = load(input1File);
ICRefMax = load(input2File);

% Call the icatb_gigicar function
[ICOutMax, TCMax] = icatb_gigicar(FmriMatr, ICRefMax);

% Save the output matrices to files
save(output1File, 'ICOutMax');
save(output2File, 'TCMax');
