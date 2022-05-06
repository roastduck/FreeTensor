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

static Opt<bool> getBoolEnv(const char *name) {
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock); // getenv is not thread safe
    char *_env = getenv(name);
    if (_env == nullptr) {
        return nullptr;
    }
    std::string env(_env);
    if (env == "true" || env == "yes" || env == "on" || env == "1") {
        return Opt<bool>::make(true);
    } else if (env == "false" || env == "no" || env == "off" || env == "0") {
        return Opt<bool>::make(false);
    } else {
        ERROR((std::string) "Value of " + name +
              " must be true/yes/on/1 or false/no/off/0");
    }
}

bool Config::prettyPrint_ = false;
bool Config::printAllId_ = false;
bool Config::werror_ = false;
bool Config::debugBinary_ = false;
Ref<Device> Config::defaultDevice_;

void Config::init() {
    Config::setPrettyPrint(isatty(fileno(stdout)));

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
