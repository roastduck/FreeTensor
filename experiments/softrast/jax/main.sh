#!/usr/bin/env bash

if [ $# != 2 ]; then
    echo "Usage: ./main.sh <cpu/gpu> <obj-file>"
    exit -1
fi

JAX_PLATFORM_NAME=$1 python3 main.py $2
