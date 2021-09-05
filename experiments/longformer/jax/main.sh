#!/usr/bin/env bash

if [ $# != 1 ]; then
    echo "Usage: ./main.sh <cpu/gpu>"
    exit -1
fi

JAX_PLATFORM_NAME=$1 python3 main.py
