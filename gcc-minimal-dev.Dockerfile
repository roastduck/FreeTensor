FROM ubuntu:22.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ python3 python3-dev python3-pip python3-venv cmake make ninja-build \
    autoconf automake libtool openjdk-11-jdk libgmp-dev

WORKDIR /opt/freetensor
COPY . .
RUN PY_BUILD_CMAKE_VERBOSE=1 pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -v -e .

WORKDIR /workspace
