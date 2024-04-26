#!/bin/bash
source ./modules
set -x
rm -f sumtimes
#mpiCC -I. -std=gnu++1y -qsmp=omp -g -O -o sumtimes ../sumtimes.cc -L${CUDA_DIR}/lib64 -lcudart
nvcc -I. -x cu -arch=sm_70 -std=c++17 -g -O2 -c ../sumtimes.cc
mpiCC -g -O -o sumtimes sumtimes.o -L${CUDA_DIR}/lib64 -lcublas -lcudart
