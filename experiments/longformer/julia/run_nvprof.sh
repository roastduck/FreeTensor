#!/usr/bin/env bash

# Use the local version of CUDA
export JULIA_CUDA_USE_BINARYBUILDER=false

# Profiling for all metrics is extremely slow, so we only profile selected metrics, which should be consistent with what we use in gather_nvprof_log.py

PROFILE_GPU=1 nvprof --profile-from-start off -m dram_read_bytes,dram_write_bytes,l2_global_load_bytes,l2_local_load_bytes,l2_global_atomic_store_bytes,l2_local_global_store_bytes,flop_count_sp,flop_count_sp_mul,flop_count_sp_fma julia gpu.jl gpu Inf 10 1 >nvprof_inference.log 2>&1
python3 ../../gather_nvprof_log.py nvprof_inference.log
