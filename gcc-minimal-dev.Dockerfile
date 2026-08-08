FROM ubuntu:22.04

ARG PYTHON_EXTRAS=""

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ python3 python3-dev python3-pip python3-venv cmake make ninja-build \
    autoconf automake libtool openjdk-11-jdk libgmp-dev

WORKDIR /opt/freetensor
COPY . .
RUN install_target="."; \
    if [ -n "${PYTHON_EXTRAS}" ]; then install_target=".[${PYTHON_EXTRAS}]"; fi; \
    PY_BUILD_CMAKE_VERBOSE=1 pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -v -e "${install_target}"

WORKDIR /workspace
