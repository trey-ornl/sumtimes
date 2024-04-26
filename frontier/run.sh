#!/bin/bash
source ./modules
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
ulimit -c 0
NODES=$1
TASKS=$(( NODES * 8 ))

ldd ./sumtimes

date
srun -t 5:00 --exclusive -n${TASKS} -N${NODES} -c7 --gpus-per-task=1 --gpu-bind=closest --unbuffered ./sumtimes
date
