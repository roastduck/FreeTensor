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

set(BUILD_SHARED_LIBS ON)

# PyBind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd-party/pybind11)

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
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd-party/z3)

# ANTLR
# 1. Runtime
set(ANTLR_BUILD_CPP_TESTS OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd-party/antlr/antlr4/runtime/Cpp)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/antlr/antlr4/runtime/Cpp/runtime/src)
# 2. Generator
find_package(Java QUIET REQUIRED COMPONENTS Runtime)
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/3rd-party/antlr/antlr-4.10.1-complete.jar)
macro(ANTLR_TARGET Name InputFile)
    set(ANTLR_ONE_VALUE_ARGS PACKAGE)
    set(ANTLR_MULTI_VALUE_ARGS COMPILE_FLAGS DEPENDS)
    cmake_parse_arguments(ANTLR_TARGET
                          "" "${ANTLR_ONE_VALUE_ARGS}" "${ANTLR_MULTI_VALUE_ARGS}" ${ARGN})
    set(ANTLR_${Name}_INPUT ${InputFile})
    set(ANTLR_${Name}_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/antlr4cpp_generated_src)
    file(MAKE_DIRECTORY ${ANTLR_${Name}_OUTPUT_DIR})
    unset(ANTLR_${Name}_CXX_OUTPUTS)
    list(APPEND ANTLR_${Name}_CXX_OUTPUTS
         ${ANTLR_${Name}_OUTPUT_DIR}/${Name}.cpp)
    list(APPEND ANTLR_${Name}_OUTPUTS
         ${ANTLR_${Name}_OUTPUT_DIR}/${Name}.h
         ${ANTLR_${Name}_OUTPUT_DIR}/${Name}.interp
         ${ANTLR_${Name}_OUTPUT_DIR}/${Name}.tokens)
    if(ANTLR_TARGET_PACKAGE)
        list(APPEND ANTLR_TARGET_COMPILE_FLAGS -package ${ANTLR_TARGET_PACKAGE})
    endif()
    list(APPEND ANTLR_${Name}_OUTPUTS ${ANTLR_${Name}_CXX_OUTPUTS})
    add_custom_command(
        OUTPUT ${ANTLR_${Name}_OUTPUTS}
        COMMAND ${Java_JAVA_EXECUTABLE} -jar ${ANTLR_EXECUTABLE}
                ${ANTLR_${Name}_INPUT}
                -o ${ANTLR_${Name}_OUTPUT_DIR}
                -no-listener
                -no-visitor
                -Dlanguage=Cpp
                -lib ${ANTLR_${Name}_OUTPUT_DIR}
                ${ANTLR_TARGET_COMPILE_FLAGS}
        DEPENDS ${ANTLR_${Name}_INPUT} ${ANTLR_TARGET_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building ${ANTLR_${Name}_INPUT} with ANTLR")
endmacro(ANTLR_TARGET)

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

# Not using CMAKE_CXX_STANDARD_REQUIRED since on GCC >= 8 we can work with C++17, let it fallback
set(CMAKE_CXX_STANDARD 20)
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

file(GLOB_RECURSE LEXERS ${CMAKE_SOURCE_DIR}/grammar/*_lexer.g)
foreach(G_FILE IN ITEMS ${LEXERS})
    get_filename_component(Name ${G_FILE} NAME_WE)
    antlr_target(${Name} ${G_FILE} PACKAGE ir)
    list(APPEND SRC ${ANTLR_${Name}_CXX_OUTPUTS})
    include_directories(${ANTLR_${Name}_OUTPUT_DIR})
    list(APPEND LEXER_OUTPUTS ${ANTLR_${Name}_OUTPUTS})
endforeach()
file(GLOB_RECURSE PARSERS ${CMAKE_SOURCE_DIR}/grammar/*_parser.g)
foreach(G_FILE IN ITEMS ${PARSERS})
    get_filename_component(Name ${G_FILE} NAME_WE)
    antlr_target(${Name} ${G_FILE} PACKAGE ir DEPENDS ${LEXER_OUTPUTS})
    list(APPEND SRC ${ANTLR_${Name}_CXX_OUTPUTS})
    include_directories(${ANTLR_${Name}_OUTPUT_DIR})
endforeach()

# Target
pybind11_add_module(ffi SHARED ${SRC})
add_dependencies(ffi ISL)
target_link_libraries(ffi PRIVATE
    ${ISL_DIR}/install/lib/libisl.so
    libz3 # See 3rd-party/z3/src/CMakeLists.txt
    antlr4_shared # See 3rd-party/antlr/antlr4/runtime/Cpp/runtime/CMakeLists.txt
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_include_directories(ffi PUBLIC ${CUDA_INCLUDE_DIRS})
