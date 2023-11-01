#!/bin/bash

module load matlab

SCRIPT=/data/users2/jwardell1/ica-torch-gica/standalone_gica_script/ica-r/matlab_work/matrixToNii.m

matlab -batch "run('${SCRIPT}')"
