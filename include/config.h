#ifndef FREE_TENSOR_CONFIG_H
#define FREE_TENSOR_CONFIG_H

#include <filesystem>
#include <vector>

#include <ref.h>

namespace freetensor {

class Target;
class Device;

/**
 * Global configurations
 *
 * All writable options with simple types can be set by environment variables
 */
class Config {
    static bool prettyPrint_;         /// Env FT_PRETTY_PRINT
    static bool printAllId_;          /// Env FT_PRINT_ALL_ID
    static bool printSourceLocation_; /// Env FT_PRINT_SOURCE_LOCATION
    static bool werror_; /// Treat warnings as errors. Env FT_WERROR
    static bool
        debugBinary_; /// Compile with `-g` at backend. Do not delete the binary
                      /// file after loaded. Env FT_DEBUG_BINARY
    static bool debugRuntimeCheck_; /// Enable runtime checks in generated code
    static bool
        debugCUDAWithUM_; /// Allocate CUDA buffers on Unified Memory, for
                          /// faster (debugging) access of GPU `Array` from CPU,
                          /// but with slower `Array` allocations and more
                          /// synchronizations. No performance effect on normal
                          /// in-kernel computations. Env FT_DEBUG_CUDA_WITH_UM
    static std::vector<std::filesystem::path>
        backendCompilerCXX_; /// Env and macro FT_BACKEND_COMPILER_CXX.
                             /// Colon-separated paths, searched from left to
                             /// right
    static std::vector<std::filesystem::path>
        backendCompilerNVCC_; /// Env and macro FT_BACKEND_COMPILER_NVCC.
                              /// Colon-separated paths, searched from left to
                              /// right

    static Ref<Target>
        defaultTarget_; /// Used for lower and codegen when target is omitted.
                        /// Initialized to CPUTarget
    static Ref<Device> defaultDevice_; /// Used to create Driver when device is
                                       /// omitted. Initialized to a CPU Device
    static std::vector<std::filesystem::path>
        runtimeDir_; /// Where to find the `runtime` directory. Macro
                     /// FT_RUNTIME_DIR. Colon-separated paths, searched from
                     /// left to right

  private:
    /**
     * Filter existing paths
     */
    static std::vector<std::filesystem::path>
    checkValidPaths(const std::vector<std::filesystem::path> &paths,
                    bool required = true);

  public:
    static void init(); /// Called in src/ffi/config.cc

    static std::string withMKL();
    static bool withCUDA();
    static bool withPyTorch();

    static void setPrettyPrint(bool pretty = true) { prettyPrint_ = pretty; }
    static bool prettyPrint() { return prettyPrint_; }

    static void setPrintAllId(bool flag = true) { printAllId_ = flag; }
    static bool printAllId() { return printAllId_; }

    static void setPrintSourceLocation(bool flag = true) {
        printSourceLocation_ = flag;
    }
    static bool printSourceLocation() { return printSourceLocation_; }

    static void setWerror(bool flag = true) { werror_ = flag; }
    static bool werror() { return werror_; }

    static void setDebugBinary(bool flag = true) { debugBinary_ = flag; }
    static bool debugBinary() { return debugBinary_; }

    static void setDebugRuntimeCheck(bool flag = true) {
        debugRuntimeCheck_ = flag;
    }
    static bool debugRuntimeCheck() { return debugRuntimeCheck_; }

    static void setDebugCUDAWithUM(bool flag = true) {
        debugCUDAWithUM_ = flag;
    }
    static bool debugCUDAWithUM() { return debugCUDAWithUM_; }

    /**
     * @brief Set the C++ compiler for CPU backend.
     *
     * @param paths : Paths to C++ compiler. Should be raw paths (unescaped).
     */
    static void
    setBackendCompilerCXX(const std::vector<std::filesystem::path> &paths) {
        backendCompilerCXX_ = checkValidPaths(paths);
    }
    static const std::vector<std::filesystem::path> &backendCompilerCXX() {
        return backendCompilerCXX_;
    }

    /**
     * @brief Set the NVCC compiler for GPU backend.
     *
     * @param paths : Paths to NVCC compiler. Should be raw paths (unescaped).
     */
    static void
    setBackendCompilerNVCC(const std::vector<std::filesystem::path> &paths) {
        backendCompilerNVCC_ = checkValidPaths(paths);
    }
    static const std::vector<std::filesystem::path> &backendCompilerNVCC() {
        return backendCompilerNVCC_;
    }

    static void setDefaultTarget(const Ref<Target> &target) {
        defaultTarget_ = target;
    }
    static Ref<Target> defaultTarget() { return defaultTarget_; }

    static void setDefaultDevice(const Ref<Device> &dev) {
        defaultDevice_ = dev;
    }
    static Ref<Device> defaultDevice() { return defaultDevice_; }

    static void setRuntimeDir(const std::vector<std::filesystem::path> &paths) {
        runtimeDir_ = checkValidPaths(paths);
    }
    static const std::vector<std::filesystem::path> &runtimeDir() {
        return runtimeDir_;
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_CONFIG_H
