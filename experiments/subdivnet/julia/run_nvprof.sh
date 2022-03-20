#!/usr/bin/env bash

# Use the local version of CUDA
export JULIA_CUDA_USE_BINARYBUILDER=false

PROFILE_GPU=1 nvprof --profile-from-start off -m all julia gpu.jl gpu Inf 10 1 >nvprof_inference.log 2>&1
python3 ../../gather_nvprof_log.py nvprof_inference.log
