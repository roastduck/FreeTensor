#include <config.h>
#include <driver/device.h>
#include <driver/target.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_config(py::module_ &m) {
    Config::init();

    m.def("with_mkl", Config::withMKL, "Check if FreeTensor is built with MKL");
    m.def("with_cuda", Config::withCUDA,
          "Check if FreeTensor is built with CUDA");
    m.def("with_pytorch", Config::withPyTorch,
          "Check if FreeTensor is built with PyTorch interface");
    m.def("set_pretty_print", Config::setPrettyPrint, "Set colored printing",
          "flag"_a = true);
    m.def("pretty_print", Config::prettyPrint,
          "Check if colored printing enabled");
    m.def("set_print_all_id", Config::setPrintAllId,
          "Print IDs of all statements in an AST", "flag"_a = true);
    m.def("print_all_id", Config::printAllId,
          "Check if printing IDs of all statements in an AST");
    m.def("set_print_source_location", Config::setPrintSourceLocation,
          "Print Python source location of all statements in an AST",
          "flag"_a = true);
    m.def(
        "print_source_location", Config::printSourceLocation,
        "Check if printing Python source location of all statements in an AST");
    m.def("fast_math", Config::fastMath,
          "Run `pass/float_simplify` optimization pass, and enable fast math "
          "on backend compilers");
    m.def("set_fast_math", Config::setFastMath,
          "Set to run `pass/float_simplify` optimization pass, and enable fast "
          "math on backend compilers (or not)",
          "flag"_a = true);
    m.def("set_werror", Config::setWerror, "Error on warning", "flag"_a = true);
    m.def("werror", Config::werror, "Check if error-on-warning enabled");
    m.def("set_debug_binary", Config::setDebugBinary,
          "Compile with `-g` at backend. FreeTensor will not delete the binary "
          "file after loading it",
          "flag"_a = true);
    m.def("debug_binary", Config::debugBinary,
          "Check if compiling binary in debug mode");
    m.def("set_debug_cuda_with_um", Config::setDebugCUDAWithUM,
          "Allocate CUDA buffers on Unified Memory, for faster (debugging) "
          "access of GPU `Array` from CPU, but with slower `Array` allocations "
          "and more synchronizations. No performance effect on normal "
          "in-kernel computations");
    m.def("debug_cuda_with_um", Config::debugCUDAWithUM,
          "Check if debugging with Unified Memory enabled");
    m.def(
        "set_backend_compiler_cxx",
        [](const std::vector<std::string> &paths) {
            auto pathsFs =
                paths | views::transform([](const std::string &path) {
                    return std::filesystem::path(path);
                });
            Config::setBackendCompilerCXX({pathsFs.begin(), pathsFs.end()});
        },
        "Set backend compiler used to compile generated C++ code, unescaped "
        "raw path expected",
        "path"_a);
    m.def(
        "backend_compiler_cxx",
        []() {
            auto &&paths = Config::backendCompilerCXX();
            return std::vector<std::string>(paths.begin(), paths.end());
        },
        "Backend compiler used to compile generated C++ code");
    m.def(
        "set_backend_compiler_nvcc",
        [](const std::vector<std::string> &paths) {
            auto pathsFs =
                paths | views::transform([](const std::string &path) {
                    return std::filesystem::path(path);
                });
            Config::setBackendCompilerNVCC({pathsFs.begin(), pathsFs.end()});
        },
        "Set backend compiler used to compile generated CUDA code, unescaped "
        "raw path expected",
        "path"_a);
    m.def(
        "backend_compiler_nvcc",
        []() {
            auto &&paths = Config::backendCompilerNVCC();
            return std::vector<std::string>(paths.begin(), paths.end());
        },
        "Backend compiler used to compile generated CUDA code");
    m.def(
        "backend_openmp",
        []() {
            auto &&paths = Config::backendOpenMP();
            return std::vector<std::string>(paths.begin(), paths.end());
        },
        "OpenMP library linked to the compiled executable");
    m.def(
        "set_backend_openmp",
        [](const std::vector<std::string> &paths) {
            auto pathsFs =
                paths | views::transform([](const std::string &path) {
                    return std::filesystem::path(path);
                });
            Config::setBackendOpenMP({pathsFs.begin(), pathsFs.end()});
        },
        "Set the OpenMP library linked to the compiled executable");
    m.def("set_default_target", Config::setDefaultTarget,
          "Set default target (internal implementation of `with Target`)",
          "target"_a);
    m.def("default_target", Config::defaultTarget,
          "Check current default target");
    m.def("set_default_device", Config::setDefaultDevice,
          "Set default device (internal implementation of `with Device`)",
          "device"_a);
    m.def("default_device", Config::defaultDevice,
          "Check current default device");
    m.def(
        "set_runtime_dir",
        [](const std::vector<std::string> &paths) {
            auto pathsFs =
                paths | views::transform([](const std::string &path) {
                    return std::filesystem::path(path);
                });
            Config::setRuntimeDir({pathsFs.begin(), pathsFs.end()});
        },
        "Set serach paths for FreeTensor runtime header files", "paths"_a);
    m.def(
        "runtime_dir",
        []() {
            auto &&paths = Config::runtimeDir();
            return std::vector<std::string>(paths.begin(), paths.end());
        },
        "Serach paths for FreeTensor runtime header files");
}

} // namespace freetensor
