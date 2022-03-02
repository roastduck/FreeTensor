#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "Usage: ./main.sh <cpu/gpu> [--warmup-repeat <NUM>] [--timing-repeat <NUM>]"
    exit -1
fi

JAX_PLATFORM_NAME=$1 python3 main.py ${@: 2}
