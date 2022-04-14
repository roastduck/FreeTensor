#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace ir {

/**
 * Global configurations
 *
 * All writable options can be set by environment variables
 */
class Config {
    static bool prettyPrint_; /// Env IR_PRETTY_PRINT
    static bool printAllId_;  /// Env IR_PRINT_ALL_ID
    static bool werror_;      /// Treat warnings as errors. Env IR_WERROR
    static bool
        debugBinary_; /// Compile with `-g` at backend. Do not delete the binary
                      /// file after loaded. Env IR_DEBUG_BINARY

  public:
    static void init(); /// Called in src/ffi/config.cc

    static std::string withMKL();

    static void setPrettyPrint(bool pretty = true) { prettyPrint_ = pretty; }
    static bool prettyPrint() { return prettyPrint_; }

    static void setPrintAllId(bool flag = true) { printAllId_ = flag; }
    static bool printAllId() { return printAllId_; }

    static void setWerror(bool flag = true) { werror_ = flag; }
    static bool werror() { return werror_; }

    static void setDebugBinary(bool flag = true) { debugBinary_ = flag; }
    static bool debugBinary() { return debugBinary_; }
};

} // namespace ir

#endif // CONFIG_H
