#ifndef FREE_TENSOR_CONFIG_H
#define FREE_TENSOR_CONFIG_H

#include <string>

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
    static bool prettyPrint_; /// Env FT_PRETTY_PRINT
    static bool printAllId_;  /// Env FT_PRINT_ALL_ID
    static bool werror_;      /// Treat warnings as errors. Env FT_WERROR
    static bool
        debugBinary_; /// Compile with `-g` at backend. Do not delete the binary
                      /// file after loaded. Env FT_DEBUG_BINARY
    static std::string backendCompilerCXX_;  /// Env FT_BACKEND_COMPILER_CXX
    static std::string backendCompilerNVCC_; /// Env FT_BACKEND_COMPILER_NVCC

    static Ref<Target> defaultTarget_; /// Used for lower and codegen when
                                       /// target is omitted. Initialized to CPU
    static Ref<Device>
        defaultDevice_; /// Used to create Array and Driver when
                        /// device is omitted. Initialized to a CPU Device

  public:
    static void init(); /// Called in src/ffi/config.cc

    static std::string withMKL();
    static bool withCUDA();

    static void setPrettyPrint(bool pretty = true) { prettyPrint_ = pretty; }
    static bool prettyPrint() { return prettyPrint_; }

    static void setPrintAllId(bool flag = true) { printAllId_ = flag; }
    static bool printAllId() { return printAllId_; }

    static void setWerror(bool flag = true) { werror_ = flag; }
    static bool werror() { return werror_; }

    static void setDebugBinary(bool flag = true) { debugBinary_ = flag; }
    static bool debugBinary() { return debugBinary_; }

    static void setBackendCompilerCXX(const std::string &path) {
        backendCompilerCXX_ = path;
    }
    static const std::string &backendCompilerCXX() {
        return backendCompilerCXX_;
    }

    static void setBackendCompilerNVCC(const std::string &path) {
        backendCompilerNVCC_ = path;
    }
    static const std::string &backendCompilerNVCC() {
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
};

} // namespace freetensor

#endif // FREE_TENSOR_CONFIG_H
