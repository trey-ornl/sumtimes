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
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes
date
sleep 5

export MPICH_GPU_COLL_STAGING_AREA_OPT=0
date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes
date
unset MPICH_GPU_COLL_STAGING_AREA_OPT
sleep 5

export MPICH_GPU_ALLREDUCE_USE_KERNEL=0
date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes
date
unset MPICH_GPU_ALLREDUCE_USE_KERNEL
sleep 5

export MPICH_GPU_NO_ASYNC_COPY=1
date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes
date
unset MPICH_GPU_NO_ASYNC_COPY
