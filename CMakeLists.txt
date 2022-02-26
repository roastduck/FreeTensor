cmake_minimum_required(VERSION 3.15) # Requried by find python
project(IR C CXX)

include(ExternalProject)

option(IR_DEBUG_LOG_NODE "Log where each AST node is created" OFF)
option(IR_DEBUG_PROFILE "Profile some heavy functions in the compiler" OFF)
option(WITH_MKL "Path to MKL (optional)" OFF)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
endif()

# PyBind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(3rd-party/pybind11)

# ISL
set(ISL_DIR ${CMAKE_SOURCE_DIR}/3rd-party/isl)
ExternalProject_add(ISL
    SOURCE_DIR ${ISL_DIR}
    BUILD_IN_SOURCE true
    INSTALL_DIR ${ISL_DIR}/install
    CONFIGURE_COMMAND ./autogen.sh && ./configure --prefix=${ISL_DIR}/install
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install)
include_directories(${ISL_DIR}/install/include)
link_directories(${ISL_DIR}/install/lib)

# Z3
set(SUBMODULE_DIR_NAME "3rd-party") # Z3 PR #4959
add_subdirectory(3rd-party/z3)

# CUDA
find_package(CUDA REQUIRED)

# THIS PROJECT
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
if(WITH_MKL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MKL=${WITH_MKL}")
endif()
if(IR_DEBUG_LOG_NODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIR_DEBUG_LOG_NODE")
endif()
if(IR_DEBUG_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIR_DEBUG_PROFILE")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=gnu++11 when on, -std=c++11 when off

add_definitions(-DIR_RUNTIME_DIR=${CMAKE_SOURCE_DIR}/runtime)

# OpenMP used to parallelize the compilation of different instances
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/pybind11/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/z3/src/api/c++)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/cppitertools)
file(GLOB_RECURSE SRC ${CMAKE_SOURCE_DIR}/src/*.cc)

# Target
pybind11_add_module(ffi SHARED ${SRC})
add_dependencies(ffi ISL)
target_link_libraries(ffi PRIVATE
    ${ISL_DIR}/install/lib/libisl.so
    libz3 # See 3rd-party/z3/src/CMakeLists.txt
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

