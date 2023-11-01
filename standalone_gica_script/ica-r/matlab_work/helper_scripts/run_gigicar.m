addpath("/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work")

% Parse arguments
input1File = getenv("inputArg1");
input2File = getenv("inputArg2");

% Load the input matrices from files
loadedData1 = load(input1File);  % Load the first data file
loadedData2 = load(input2File);  % Load the second data file

% Access the data from the loaded structure
FmriMatr = loadedData1.subjectData;  % Assuming the field name is "subjectData"
ICRefMax = loadedData2.groupData;    % Assuming the field name is "groupData"

% Call the icatb_gigicar function
[ICOutMax, TCMax] = icatb_gigicar(FmriMatr, ICRefMax);

% Save the output matrices to files
save('ICOutMax.mat', 'ICOutMax', '-double');
save('TCMax.mat', 'TCMax', '-double');
