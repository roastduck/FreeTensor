#!/usr/bin/env bash

if [[ -z "${TVM_HOME}" ]]; then
    echo "Please specify TVM_HOME"
    exit -1
fi

PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} python3 part1.py $1
PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} python3 part2.py $@
