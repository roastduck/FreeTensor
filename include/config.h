#ifndef FREE_TENSOR_CONFIG_H
#define FREE_TENSOR_CONFIG_H

#include <string>

namespace freetensor {

/**
 * Global configurations
 *
 * All writable options can be set by environment variables
 */
class Config {
    static bool prettyPrint_; /// Env FT_PRETTY_PRINT
    static bool printAllId_;  /// Env FT_PRINT_ALL_ID
    static bool werror_;      /// Treat warnings as errors. Env FT_WERROR
    static bool
        debugBinary_; /// Compile with `-g` at backend. Do not delete the binary
                      /// file after loaded. Env FT_DEBUG_BINARY

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
};

} // namespace freetensor

#endif // FREE_TENSOR_CONFIG_H
