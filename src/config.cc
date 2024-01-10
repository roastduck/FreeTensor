#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <unistd.h>

#include <config.h>
#include <container_utils.h>
#include <driver/device.h>
#include <serialize/to_string.h>

namespace freetensor {

namespace fs = std::filesystem;

/**
 * Make paths from a colon-separated string
 */
static std::vector<fs::path> makePaths(const std::string &str) {
    std::vector<std::string> ret(1, "");
    for (char c : str) {
        if (c == ':') {
            ret.emplace_back();
        } else {
            ret.back().push_back(c);
        }
    }
    return std::vector<fs::path>(ret.begin(), ret.end());
}

/**
 * Return paths of a filename in some directories
 */
static std::vector<fs::path> fileInDirs(const std::string &filename,
                                        const std::vector<fs::path> &dirs) {
    std::vector<fs::path> ret;
    ret.reserve(dirs.size());
    for (auto &&dir : dirs) {
        ret.emplace_back(dir / filename);
    }
    return ret;
}

static std::optional<std::string> getStrEnv(const char *name) {
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock); // getenv is not thread safe
    char *env = getenv(name);
    if (env == nullptr) {
        return std::nullopt;
    } else {
        return std::make_optional<std::string>(env);
    }
}

static std::string getStrEnvRequired(const char *name) {
    if (auto &&ret = getStrEnv(name); ret.has_value()) {
        return *ret;
    } else {
        ERROR(FT_MSG << "Environment varialbe " << name << " not found");
    }
}

static std::optional<bool> getBoolEnv(const char *name) {
    if (auto _env = getStrEnv(name); _env.has_value()) {
        auto &&env = tolower(*_env);
        if (env == "true" || env == "yes" || env == "on" || env == "1") {
            return true;
        } else if (env == "false" || env == "no" || env == "off" ||
                   env == "0") {
            return false;
        } else {
            ERROR(FT_MSG << "Value of " << name
                         << " must be true/yes/on/1 or false/no/off/0 (case "
                            "insensitive)");
        }
    } else {
        return std::nullopt;
    }
}

bool Config::prettyPrint_ = false;
bool Config::printAllId_ = false;
bool Config::printSourceLocation_ = false;
bool Config::fastMath_ = true;
bool Config::werror_ = false;
bool Config::debugBinary_ = false;
bool Config::debugRuntimeCheck_ = false;
bool Config::debugCUDAWithUM_ = false;
std::vector<fs::path> Config::backendCompilerCXX_;
std::vector<fs::path> Config::backendCompilerNVCC_;
std::vector<fs::path> Config::backendOpenMP_;
Ref<Target> Config::defaultTarget_;
Ref<Device> Config::defaultDevice_;
std::vector<fs::path> Config::runtimeDir_;

std::vector<fs::path>
Config::checkValidPaths(const std::vector<fs::path> &paths, bool required) {
    auto ret =
        filter(paths, static_cast<bool (*)(const fs::path &)>(fs::exists));
    if (required && ret.empty()) {
        ERROR(FT_MSG << paths << " are not valid paths");
    }
    return ret;
}

void Config::init() {
    Config::setPrettyPrint(isatty(fileno(stdout)));
#ifdef FT_BACKEND_COMPILER_CXX
    Config::setBackendCompilerCXX(
        cat(makePaths(FT_BACKEND_COMPILER_CXX),
            fileInDirs("g++", makePaths(getStrEnvRequired("PATH")))));
#else
    Config::setBackendCompilerCXX(
        fileInDirs("g++", makePaths(getStrEnvRequired("PATH"))));
#endif
#ifdef FT_WITH_CUDA
#ifdef FT_BACKEND_COMPILER_NVCC
    Config::setBackendCompilerNVCC(
        cat(makePaths(FT_BACKEND_COMPILER_NVCC),
            fileInDirs("nvcc", makePaths(getStrEnvRequired("PATH")))));
#else
    Config::setBackendCompilerNVCC(
        fileInDirs("nvcc", makePaths(getStrEnvRequired("PATH"))));
#endif // FT_BACKEND_COMPILER_NVCC
#endif // FT_WITH_CUDA
#ifdef FT_BACKEND_OPENMP
    Config::setBackendOpenMP(makePaths(FT_BACKEND_OPENMP));
#endif

    if (auto flag = getBoolEnv("FT_PRETTY_PRINT"); flag.has_value()) {
        Config::setPrettyPrint(*flag);
    }
    if (auto flag = getBoolEnv("FT_PRINT_ALL_ID"); flag.has_value()) {
        Config::setPrintAllId(*flag);
    }
    if (auto flag = getBoolEnv("FT_PRINT_SOURCE_LOCATION"); flag.has_value()) {
        Config::setPrintSourceLocation(*flag);
    }
    if (auto flag = getBoolEnv("FT_FAST_MATH"); flag.has_value()) {
        Config::setFastMath(*flag);
    }
    if (auto flag = getBoolEnv("FT_WERROR"); flag.has_value()) {
        Config::setWerror(*flag);
    }
    if (auto flag = getBoolEnv("FT_DEBUG_BINARY"); flag.has_value()) {
        Config::setDebugBinary(*flag);
    }
    if (auto flag = getBoolEnv("FT_DEBUG_RUNTIME_CHECK"); flag.has_value()) {
        Config::setDebugRuntimeCheck(*flag);
    }
    if (auto flag = getBoolEnv("FT_DEBUG_CUDA_WITH_UM"); flag.has_value()) {
        Config::setDebugCUDAWithUM(*flag);
    }
    if (auto path = getStrEnv("FT_BACKEND_COMPILER_CXX"); path.has_value()) {
        Config::setBackendCompilerCXX(makePaths(*path));
    }
#ifdef FT_WITH_CUDA
    if (auto path = getStrEnv("FT_BACKEND_COMPILER_NVCC"); path.has_value()) {
        Config::setBackendCompilerNVCC(makePaths(*path));
    }
#endif // FT_WITH_CUDA
    if (auto &&path = getStrEnv("FT_BACKEND_OPENMP"); path.has_value()) {
        Config::setBackendOpenMP(makePaths(*path));
    }
    auto device = Ref<Device>::make(TargetType::CPU);
    Config::setDefaultDevice(device);
    Config::setDefaultTarget(device->target());

#ifdef FT_RUNTIME_DIR
    Config::setRuntimeDir(makePaths(FT_RUNTIME_DIR));
#else
#error "FT_RUNTIME_DIR has to be defined"
#endif
}

bool Config::withMKL() {
#ifdef FT_WITH_MKL
    return true;
#else
    return false;
#endif
}

bool Config::withCUDA() {
#ifdef FT_WITH_CUDA
    return true;
#else
    return false;
#endif
}

bool Config::withPyTorch() {
#ifdef FT_WITH_PYTORCH
    return true;
#else
    return false;
#endif
}

} // namespace freetensor
