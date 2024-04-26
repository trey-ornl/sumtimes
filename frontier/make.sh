#!/bin/bash
source ./modules
module -t list
set -x
rm -f sumtimes
hipcc -g -O -I${CRAY_MPICH_DIR}/include -o sumtimes ../sumtimes.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} -lhipblas
