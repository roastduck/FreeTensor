cmake_minimum_required(VERSION 3.15) # Requried by find python
project(IR C CXX)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=gnu++11 when on, -std=c++11 when off

find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(3rd-party/pybind11)

include_directories(include)
include_directories(3rd-party/pybind11/include)
file(GLOB_RECURSE SRC src/*.cc)

# Target
pybind11_add_module(ffi SHARED ${SRC})

