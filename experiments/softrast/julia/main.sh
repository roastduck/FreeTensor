#!/usr/bin/env bash

if [ $# != 2 ]; then
    echo "Usage: ./main.sh <cpu/gpu> <obj-file>"
    exit -1
fi

if [ $1 == 'cpu' ]; then
    julia cpu.jl $2
fi
