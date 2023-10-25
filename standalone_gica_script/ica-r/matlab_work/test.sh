#!/bin/bash

module load matlab

FOO=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/foo.txt
BAR=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/bar.txt
SCRIPT=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/test.m

matlab -batch "setenv('inputArg1', '${FOO}'); setenv('inputArg2', '${BAR}'); run('${SCRIPT}')"
