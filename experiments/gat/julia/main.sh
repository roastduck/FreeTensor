#!/usr/bin/env bash

if [ $# != 1 ]; then
    echo "Usage: ./main.sh <cpu/gpu>"
    exit -1
fi

if [ $1 == 'gpu' ]; then
    JULIA_CUDA_USE_BINARYBUILDER=false julia ./gpu.jl
fi
