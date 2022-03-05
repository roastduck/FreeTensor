#!/usr/bin/env bash

# Use the local version of CUDA
export JULIA_CUDA_USE_BINARYBUILDER=false

if [ $1 == 'cpu' ]; then
    threads=`cat /proc/cpuinfo | grep "processor" | wc -l`
    JULIA_NUM_THREADS=$threads julia cpu.jl $@
elif [ $1 == 'gpu' ]; then
    JULIA_CUDA_USE_BINARYBUILDER=false julia ./gpu.jl $@
fi
