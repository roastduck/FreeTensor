#!/usr/bin/env bash

if [[ -z "${TVM_HOME}" ]]; then
    echo "Please specify TVM_HOME"
    exit -1
fi

PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} python3 main.py $@
