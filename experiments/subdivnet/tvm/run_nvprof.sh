#!/usr/bin/env bash

if [[ -z "${TVM_HOME}" ]]; then
    echo "Please specify TVM_HOME"
    exit -1
fi

PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} nvprof --profile-from-start off -m all python3 main.py gpu --profile-gpu --timing-repeat 1 $@ >nvprof_inference.log 2>&1
python3 ../../gather_nvprof_log.py nvprof_inference.log
