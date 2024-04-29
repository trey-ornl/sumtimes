#!/bin/bash
#SBATCH -t 5:00 -J sumtimes
source ./modules
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_CXI_COUNTER_REPORT=2
export MPICH_ENV_DISPLAY=1
#ulimit -c unlimited
ulimit -c 0
NODES=${SLURM_JOB_NUM_NODES}
TASKS=$(( NODES * 8 ))

ldd ./sumtimes

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes 0 0 1
date
sleep 5

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes 0 0 32
date
sleep 5

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes 0 0 1024
date
sleep 5

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes 0 0 32768
date
sleep 5

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes 0 0 1048576
date
sleep 5

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes 0 0 33554432
date
sleep 5

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes 0 0 1073741824
date
