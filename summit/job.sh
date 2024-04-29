#!/bin/bash
#BSUB -alloc_flags smt1 
#BSUB -W 7 
#BSUB -J sumtimes

source ./modules
set -x
ulimit -c 0

ldd ./sumtimes

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./sumtimes 0 0 268435456
date
