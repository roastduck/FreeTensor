#!/usr/bin/env bash

PYTHONPATH=../../../python:../../../build:$PYTHONPATH nvprof -m all python3 main.py gpu --infer-only --warmup-repeat 0 --timing-repeat 1 >nvprof_inference.log 2>&1
python3 ../../gather_nvprof_log.py nvprof_inference.log
