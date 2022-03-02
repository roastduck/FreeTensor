#!/usr/bin/env bash

# Use the local version of CUDA
export JULIA_CUDA_USE_BINARYBUILDER=false

julia why_julia_gpu_is_wrong.jl $@
