cmake_minimum_required(VERSION 3.15) # Requried by find python
project(FREE_TENSOR C CXX)

include(ExternalProject)

option(FT_DEBUG_BLAME_AST "Log where each AST node is created" OFF)
option(FT_DEBUG_PROFILE "Profile some heavy functions in the compiler" OFF)
option(FT_DEBUG_SANITIZE "Build with GCC sanitizer. Can be set to OFF or a sanitizer name (e.g. address)" OFF)
option(FT_WITH_CUDA "Build with CUDA (ON / OFF)" ON)
option(FT_WITH_MKL "Build with MKL (ON / Path to MKL / OFF)" OFF)
option(FT_WITH_PYTORCH "Build with PyTorch interface" OFF)
option(FT_COMPILER_PORTABLE "Do not build FreeTensor itself and its dependencies with non-portable instructions" OFF)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
endif()

# Check compiler version
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
        message(FATAL_ERROR "The minimum required version for GCC is 11")
    endif()
endif()

# Check whther 3rd-party modules exist
foreach(MODULE IN ITEMS antlr/antlr4 range-v3 cuda-samples isl pybind11 z3)
    file(GLOB RESULT ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/${MODULE}/*)
    list(LENGTH RESULT RES_LEN)
    if(RES_LEN EQUAL 0)
        message(FATAL_ERROR "3rd-party module ${MODULE} not found, run `git submodule update --init --recursive` to clone 3rd-party modules")
    endif()
endforeach()

# Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# PyBind11
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/pybind11)

# ISL
set(ISL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/isl)
if (FT_COMPILER_PORTABLE)
    set(ISL_ENABLE_PORTABLE "--enable-portable-binary")
endif()
ExternalProject_add(ISL
    SOURCE_DIR ${ISL_DIR}
    BUILD_IN_SOURCE true
    INSTALL_DIR ${ISL_DIR}/install
    CONFIGURE_COMMAND ./autogen.sh && ./configure --prefix=${ISL_DIR}/install ${ISL_ENABLE_PORTABLE}
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${ISL_DIR}/install/lib/libisl.so)
add_library(isl::isl INTERFACE IMPORTED GLOBAL)
make_directory(${ISL_DIR}/install/include)
target_include_directories(isl::isl INTERFACE ${ISL_DIR}/install/include)
target_link_libraries(isl::isl INTERFACE ${ISL_DIR}/install/lib/libisl.so)
add_dependencies(isl::isl ISL)

# Z3
set(SUBMODULE_DIR_NAME "3rd-party") # Z3 PR #4959
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/z3)

# ANTLR
# 1. Runtime
set(ANTLR_BUILD_CPP_TESTS OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/antlr/antlr4/runtime/Cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/antlr/antlr4/runtime/Cpp/runtime/src)
# 2. Generator
find_package(Java QUIET REQUIRED COMPONENTS Runtime)
set(ANTLR_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/antlr/antlr-4.11.1-complete.jar)
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
if(FT_WITH_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DFT_WITH_CUDA)
endif()

# PyTorch
if(FT_WITH_PYTORCH)
    # https://pytorch.org/cppdocs/installing.html
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch;print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PYTORCH_CMAKE_PATH)
    string(STRIP "${PYTORCH_CMAKE_PATH}" PYTORCH_CMAKE_PATH)
    message(STATUS "Looking for PyTorch CMake package in ${PYTORCH_CMAKE_PATH}")
    list(PREPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PATH})
    find_package(Torch REQUIRED)
    add_definitions(-DFT_WITH_PYTORCH)

    # find_package(Torch) does not expose PyTorch's Python binding, but we need it
    list(APPEND TORCH_LIBRARIES "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

    # Use the C++11 (gcc >= 5) ABI
    # PyTorch's pip release uses the old C++ ABI. When linking PyTorch in CMake, it automatically
    # sets FreeTensor to use the old ABI as well. The only difference between the two ABIs are
    # std::string and std::list (https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).
    # We don't pass std::string or std::list from or to PyTorch, so we can keep FreeTensor
    # to use the new ABI.
    foreach(TORCH_LIBRARY IN ITEMS ${TORCH_LIBRARIES})
        if(TARGET ${TORCH_LIBRARY})
            get_target_property(DEF ${TORCH_LIBRARY} INTERFACE_COMPILE_OPTIONS)
            message(STATUS "Checking ${TORCH_LIBRARY}: ${DEF}")
            list(REMOVE_ITEM DEF "-D_GLIBCXX_USE_CXX11_ABI=0")
            set_property(TARGET ${TORCH_LIBRARY} PROPERTY INTERFACE_COMPILE_OPTIONS ${DEF})
        endif()
    endforeach()

    # Ensure CUDA used by PyTorch is the same with CUDA used by FreeTensor, or it may fail silently
    if(FT_WITH_CUDA)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import torch;print(torch.version.cuda)"
            OUTPUT_VARIABLE PYTORCH_CUDA_VERSION)
        string(STRIP "${PYTORCH_CUDA_VERSION}" PYTORCH_CUDA_VERSION)
        if(NOT ${PYTORCH_CUDA_VERSION} STREQUAL ${CUDA_VERSION})
            message(FATAL_ERROR "CUDA version used in PyTorch (${PYTORCH_CUDA_VERSION}) should match CUDA version used in FreeTensor (${CUDA_VERSION})")
        endif()
    endif()
endif()

# THIS PROJECT
# All paths passed to pre-defined macros should be quoted here. CMake add_definitions
# kindly preserves the quotes through escaping. Once expanded by C preprocessor, they
# become string literals in C/C++ code and we are safe to use them.
set(CMAKE_CXX_FLAGS "-Wall -Werror ${CMAKE_CXX_FLAGS}")  # Prepend so users can still use "-DCMAKE_CXX_FLAGS" to override flags
if(NOT FT_COMPILER_PORTABLE)
    set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()
if(FT_WITH_MKL)
    add_definitions(-DFT_WITH_MKL)
    # Libraries are selected by MKL Link Line Advisor
    # Link statically, or there will be dlopen issues
    if(IS_DIRECTORY "${FT_WITH_MKL}")  # IS_DIRECTORY's argument should be a string
        set(FT_MKL_INCLUDE "${FT_WITH_MKL}/include")
        set(FT_MKL_LIBMKL_INTEL_LP64 "${FT_WITH_MKL}/lib/intel64/libmkl_intel_lp64.a")
        set(FT_MKL_LIBMKL_GNU_THREAD "${FT_WITH_MKL}/lib/intel64/libmkl_gnu_thread.a")
        set(FT_MKL_LIBMKL_CORE "${FT_WITH_MKL}/lib/intel64/libmkl_core.a")
    else()
        find_path(FT_MKL_INCLUDE mkl.h PATH_SUFFIXES mkl REQUIRED)
        find_library(FT_MKL_LIBMKL_INTEL_LP64 libmkl_intel_lp64.a REQUIRED)
        find_library(FT_MKL_LIBMKL_GNU_THREAD libmkl_gnu_thread.a REQUIRED)
        find_library(FT_MKL_LIBMKL_CORE libmkl_core.a REQUIRED)
    endif()
    add_definitions(-DFT_MKL_INCLUDE="${FT_MKL_INCLUDE}")
    add_definitions(-DFT_MKL_LIBMKL_INTEL_LP64="${FT_MKL_LIBMKL_INTEL_LP64}")
    add_definitions(-DFT_MKL_LIBMKL_GNU_THREAD="${FT_MKL_LIBMKL_GNU_THREAD}")
    add_definitions(-DFT_MKL_LIBMKL_CORE="${FT_MKL_LIBMKL_CORE}")
endif()
if(FT_DEBUG_BLAME_AST)
    add_definitions(-DFT_DEBUG_BLAME_AST)
endif()
if(FT_DEBUG_PROFILE)
    add_definitions(-DFT_DEBUG_PROFILE)
endif()
if(FT_DEBUG_SANITIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=${FT_DEBUG_SANITIZE}")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=gnu++20 when on, -std=c++20 when off

add_definitions(-DFT_RUNTIME_DIR="${CMAKE_CURRENT_SOURCE_DIR}/runtime:${CMAKE_INSTALL_PREFIX}/share/runtime_include")
add_definitions(-DFT_BACKEND_COMPILER_CXX="${CMAKE_CXX_COMPILER}")
if(FT_WITH_CUDA)
    add_definitions(-DFT_BACKEND_COMPILER_NVCC="${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()

# OpenMP used to parallelize the compilation of different instances
find_package(OpenMP REQUIRED)
list(JOIN OpenMP_CXX_LIBRARIES ":" OMP_LIBS_COLON_SEPARATED)
add_definitions(-DFT_BACKEND_OPENMP="${OMP_LIBS_COLON_SEPARATED}")

file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

file(GLOB_RECURSE LEXERS ${CMAKE_CURRENT_SOURCE_DIR}/grammar/*_lexer.g)
set(ANTLR_INCLUDES "")
foreach(G_FILE IN ITEMS ${LEXERS})
    get_filename_component(Name ${G_FILE} NAME_WE)
    antlr_target(${Name} ${G_FILE} PACKAGE freetensor)
    list(APPEND SRC ${ANTLR_${Name}_CXX_OUTPUTS})
    list(APPEND ANTLR_INCLUDES ${ANTLR_${Name}_OUTPUT_DIR})
    list(APPEND LEXER_OUTPUTS ${ANTLR_${Name}_OUTPUTS})
endforeach()
file(GLOB_RECURSE PARSERS ${CMAKE_CURRENT_SOURCE_DIR}/grammar/*_parser.g)
foreach(G_FILE IN ITEMS ${PARSERS})
    get_filename_component(Name ${G_FILE} NAME_WE)
    antlr_target(${Name} ${G_FILE} PACKAGE freetensor DEPENDS ${LEXER_OUTPUTS})
    list(APPEND SRC ${ANTLR_${Name}_CXX_OUTPUTS})
    list(APPEND ANTLR_INCLUDES ${ANTLR_${Name}_OUTPUT_DIR})
endforeach()

# Target
add_library(freetensor SHARED ${SRC})
add_dependencies(freetensor ISL)
target_link_libraries(freetensor PUBLIC
    isl::isl
    libz3 # See 3rd-party/z3/src/CMakeLists.txt
    antlr4_shared # See 3rd-party/antlr/antlr4/runtime/Cpp/runtime/CMakeLists.txt
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_include_directories(freetensor PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/z3/src/api/c++
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/range-v3/include)
target_include_directories(freetensor PRIVATE ${ANTLR_INCLUDES})

file(GLOB_RECURSE FFI_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ffi/*.cc)
pybind11_add_module(freetensor_ffi SHARED ${FFI_SRC})
# PyTorch libraries needs to be linked `--as-needed`. This enables loading FreeTensor on a system
# without CUDA driver (but not running CUDA functions), even if `FT_WITH_CUDA=ON`.
list(JOIN CMAKE_CXX_LINKER_WRAPPER_FLAG " " CXX_LINKER_WRAPPER_FLAG_STR) # May be one or more arguments, ending with " " or not
set(AS_NEEDED_FLAG "${CXX_LINKER_WRAPPER_FLAG_STR}--as-needed")
target_link_libraries(freetensor_ffi PRIVATE freetensor ${AS_NEEDED_FLAG} ${TORCH_LIBRARIES})
target_include_directories(freetensor_ffi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ffi)

# Pybind11 stubgen
find_program(PYBIND11_STUBGEN pybind11-stubgen)
if (PYBIND11_STUBGEN AND NOT FT_DEBUG_SANITIZE)
    # Disable stubgen when building with a sanitizer, beacuse using such a library from Python requires preloading libasan.so
    message(STATUS "Enabling pybind11-stubgen step to generate typing stub for IDEs.")
    add_custom_command(
        TARGET freetensor_ffi POST_BUILD
        COMMAND env PYTHONPATH=.:$ENV{PYTHONPATH} ${PYBIND11_STUBGEN} freetensor_ffi --no-setup-py --root-module-suffix "" --ignore-invalid signature
        VERBATIM
    )
endif()

# Install
# Now we install freetensor_ffi to `???/lib/python3.10/dist-packages/freetensor` but others to CMAKE_INSTALL_PREFIX.
# FIXME: We should install everything to `???/lib/python3.10/dist-packages/freetensor`
set_target_properties(freetensor freetensor_ffi PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set_target_properties(freetensor_ffi PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE) # Required for PyTorch
install(DIRECTORY ${ISL_DIR}/install/lib/ DESTINATION lib) # Trailing slash needed
install(TARGETS freetensor LIBRARY DESTINATION lib)
if (PY_BUILD_CMAKE_MODULE_NAME) # Install to `???/lib/python3.10/dist-packages/freetensor`
    install(TARGETS freetensor_ffi LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT python_module)
else()
    install(TARGETS freetensor_ffi LIBRARY DESTINATION lib)
endif()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include/freetensor) # Trailing slash after the source path is needed
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ DESTINATION share/runtime_include)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/mdspan/ DESTINATION share/3rd-party/mdspan)
