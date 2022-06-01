#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unistd.h>

#include <config.h>
#include <driver/device.h>
#include <except.h>
#include <opt.h>

#define NAME_(macro) #macro
#define NAME(macro) NAME_(macro)

namespace freetensor {

static Opt<std::string> getStrEnv(const char *name) {
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock); // getenv is not thread safe
    char *env = getenv(name);
    if (env == nullptr) {
        return nullptr;
    } else {
        return Opt<std::string>::make(env);
    }
}

static Opt<bool> getBoolEnv(const char *name) {
    if (auto _env = getStrEnv(name); _env.isValid()) {
        auto &&env = *_env;
        if (env == "true" || env == "yes" || env == "on" || env == "1") {
            return Opt<bool>::make(true);
        } else if (env == "false" || env == "no" || env == "off" ||
                   env == "0") {
            return Opt<bool>::make(false);
        } else {
            ERROR((std::string) "Value of " + name +
                  " must be true/yes/on/1 or false/no/off/0");
        }
    } else {
        return nullptr;
    }
}

bool Config::prettyPrint_ = false;
bool Config::printAllId_ = false;
bool Config::werror_ = false;
bool Config::debugBinary_ = false;
std::string Config::backendCompilerCXX_;
std::string Config::backendCompilerNVCC_;
Ref<Target> Config::defaultTarget_;
Ref<Device> Config::defaultDevice_;

void Config::init() {
    Config::setPrettyPrint(isatty(fileno(stdout)));
#ifdef FT_BACKEND_COMPILER_CXX
    Config::setBackendCompilerCXX(NAME(FT_BACKEND_COMPILER_CXX));
#endif
#ifdef FT_BACKEND_COMPILER_NVCC
    Config::setBackendCompilerNVCC(NAME(FT_BACKEND_COMPILER_NVCC));
#endif

    if (auto flag = getBoolEnv("FT_PRETTY_PRINT"); flag.isValid()) {
        Config::setPrettyPrint(*flag);
    }
    if (auto flag = getBoolEnv("FT_PRINT_ALL_ID"); flag.isValid()) {
        Config::setPrintAllId(*flag);
    }
    if (auto flag = getBoolEnv("FT_WERROR"); flag.isValid()) {
        Config::setWerror(*flag);
    }
    if (auto flag = getBoolEnv("FT_DEBUG_BINARY"); flag.isValid()) {
        Config::setDebugBinary(*flag);
    }
    if (auto path = getStrEnv("FT_BACKEND_COMPILER_CXX"); path.isValid()) {
        Config::setBackendCompilerCXX(*path);
    }
    if (auto path = getStrEnv("FT_BACKEND_COMPILER_NVCC"); path.isValid()) {
        Config::setBackendCompilerNVCC(*path);
    }
    Config::setDefaultTarget(Ref<CPU>::make());
    Config::setDefaultDevice(Ref<Device>::make(Ref<CPU>::make()));
}

std::string Config::withMKL() {
#ifdef FT_WITH_MKL
    return NAME(FT_WITH_MKL);
#else
    return "";
#endif
}

bool Config::withCUDA() {
#ifdef FT_WITH_CUDA
    return true;
#else
    return false;
#endif
}

} // namespace freetensor
