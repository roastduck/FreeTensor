#!/usr/bin/env bash

if [ $# != 2 ]; then
    echo "Usage: ./main.sh <cpu/gpu> <obj-file>"
    exit -1
fi

if [ $1 == 'cpu' ]; then
    threads=`cat /proc/cpuinfo | grep "processor" | wc -l`
    JULIA_NUM_THREADS=$threads julia cpu.jl $2
fi
