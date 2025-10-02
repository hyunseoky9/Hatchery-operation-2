#!/bin/bash

#SBATCH --job-name=TD3      ## Name of the job
#SBATCH --time=24:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8      ## The number of threads the code will use
#SBATCH --mem-per-cpu=2G     ## Real memory(MB) per CPU required by the job.
ARG=${1:-1}
LOGFILE="TD3_${ARG}_${SLURM_JOB_ID}.out"
exec >"$LOGFILE" 2>&1              # send all output (stdout+stderr) to the log

## not using the default python
module purge

## Execute the python script and pass the argument/input '90'
source ~/miniconda3/bin/activate py310
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

srun python TD3_env3_x_main_HPC.py "$ARG"