#!/usr/bin/env bash

if [ $# != 1 ]; then
    echo "Usage: ./main.sh <cpu/gpu>"
    exit -1
fi

if [ $1 == 'cpu' ]; then
    julia cpu.jl
fi
