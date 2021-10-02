source /opt/spack/share/spack/setup-env.sh
source /home/bruce/install/miniconda3/bin/activate IR
spack load cmake@3.18.4
spack load gcc@9.2.0
spack load cuda@11.1.0
cd /home/bruce/hpc/IR/build
cmake ..
make -j32
cd /home/bruce/hpc/IR/experiments/subdivnet/ours
# PYTHONPATH=../../../python:../../../build:$PYTHONPATH LD_PRELOAD=`gcc -print-file-name=libasan.so` python3 main.py $@
PYTHONPATH=../../../python:../../../build:$PYTHONPATH python3 main.py $@

