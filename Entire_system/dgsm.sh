#!/bin/bash
#PBS -N dgsm_rest_50
#PBS -l select=1:ncpus=128:mem=64gb
#PBS -l walltime=3:00:00
#PBS -o O_Logs -e E_Logs
#PBS -J 0-3

cd $PBS_O_WORKDIR || exit 1

echo "Job ID: ${PBS_JOBID}"
echo "Array index: ${PBS_ARRAY_INDEX}"
echo "Started at: $(date)"

# Prevent oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Array slicing settings
TASK_ID=${PBS_ARRAY_INDEX}
N_JOBS=128
BASEPOINTS_PER_JOB=125

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate py312 || exit 1
python -u Samples_for_DGSM_HPC.py \
    --task-id "${TASK_ID}" \
    --n-jobs "${N_JOBS}" \
    --basepoints-per-job "${BASEPOINTS_PER_JOB}" \
    2> >(grep -v "ResourceTracker\|No child processes\|resource_tracker\|__del__" >&2)


echo "Finished at: $(date)"