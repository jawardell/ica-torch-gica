export PATH=/data/users2/jwardell1/miniconda3/bin:$PATH

source /data/users2/jwardell1/miniconda3/etc/profile.d/conda.sh

conda activate /data/users2/jwardell1/miniconda3/envs/ica-torch


export ICAR_DIR=/data/users2/jwardell1/ica-torch-gica/sa_script_work/ica-r/python_work

export PATHS_FILE=${ICAR_DIR}/datasets/fbirn-project/paths

export SLURM_ARRAY_TASK_ID=0

export IFS=$'\n'
export paths_array=($(cat ${PATHS_FILE}))
export sub_ix=$(( 5*$SLURM_ARRAY_TASK_ID ))
export func_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 1 ))
export out_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 2 ))
export mask_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 3 ))
export template_ix=$(( 5*$SLURM_ARRAY_TASK_ID + 4 ))

export sub_id=${paths_array[${sub_ix}]} 
export func_file=${paths_array[${func_ix}]} 
export out_dir=${paths_array[${out_ix}]} 
export mask_file=${paths_array[${mask_ix}]} 
export template_file=${paths_array[${template_ix}]}


#resample data to neuromark template
prevdir=`pwd`

module load afni


cd $out_dir
3dresample -master ${template_file} -prefix func_resampled.nii.gz -input ${func_file}
export func_file=${out_dir}/func_resampled.nii.gz

export mask_dir=`dirname $mask_file`
cd $mask_dir
3dresample -master ${template_file} -prefix mask_resampled.nii -input ${mask_file}
export mask_file=${mask_dir}/mask_resampled.nii

cd $prevdir
#python ${ICAR_DIR}/gigicar.py $sub_id $func_file $out_dir $mask_file $template_file

#cd $mask_dir
#rm mask_resampled.nii

#cd $out_dir
#rm func_resampled.nii.gz
