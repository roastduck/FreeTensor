#!/usr/bin/env bash

# Profiling JAX for all metrics is extremely slow, so we only profile selected metrics, which should be consistent with what we use in gather_nvprof_log.py
# IMPORTANT! We observed that JAX or NVPROF will write many 26G files to /tmp, so please mount /tmp as a ramdisk

JAX_PLATFORM_NAME=gpu nvprof --profile-from-start off -m dram_read_bytes,dram_write_bytes,l2_global_load_bytes,l2_local_load_bytes,l2_global_atomic_store_bytes,l2_local_global_store_bytes,flop_count_sp,flop_count_sp_mul,flop_count_sp_fma python3 main.py --profile-gpu --timing-repeat 1 2>&1 | tee nvprof_inference.log
python3 ../../gather_nvprof_log.py nvprof_inference.log
