FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG PYTHON_EXTRAS=""

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ python3 python3-dev python3-pip python3-venv cmake make ninja-build \
    autoconf automake libtool openjdk-11-jdk libgmp-dev \
    libmkl-dev

RUN pip3 install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple pip # We need `-C` from pip

WORKDIR /opt/freetensor
COPY . .
RUN install_target="."; \
    if [ -n "${PYTHON_EXTRAS}" ]; then install_target=".[${PYTHON_EXTRAS}]"; fi; \
    PY_BUILD_CMAKE_VERBOSE=1 python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -v -e "${install_target}" \
    -C--local=with-cuda.toml \
    -C--local=with-mkl.toml

WORKDIR /workspace
