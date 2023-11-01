% Parse arguments
inputArg1 = getenv("inputArg1");
inputArg2 = getenv("inputArg2");

% Display the arguments
disp(["Argument 1: " inputArg1]);
disp(["Argument 2: " inputArg2]);

% Read and print the content of the files
fileContent1 = fileread(inputArg1);
fileContent2 = fileread(inputArg2);

disp(['Content of ' inputArg1]);
disp(fileContent1);

disp(['Content of ' inputArg2]);
disp(fileContent2);

