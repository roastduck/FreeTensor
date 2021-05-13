#include <chrono>
#include <cstdio>  // remove
#include <cstdlib> // mkdtemp, system
#include <cstring> // memset
#include <dlfcn.h> // dlopen
#include <fstream>
#include <sys/stat.h> // mkdir
#include <unistd.h>   // rmdir

#include <driver.h>
#include <driver/gpu.h>
#include <except.h>

#define NAME_(macro) #macro
#define NAME(macro) NAME_(macro)

namespace ir {

Driver::Driver(const std::string &src,
               const std::vector<std::string> &paramNames, const Device &dev)
    : src_(src), params_(paramNames.size(), nullptr), dev_(dev) {
    name2param_.reserve(paramNames.size());
    for (size_t i = 0, iEnd = paramNames.size(); i < iEnd; i++) {
        name2param_[paramNames[i]] = i;
    }
    buildAndLoad();
}

void Driver::buildAndLoad() {
    std::string home = getenv("HOME");
    mkdir((home + "/.ir").c_str(), 0755);
    std::string path_string = home + "/.ir/XXXXXX";
    char path[64];
    ASSERT(path_string.size() < 64);
    strncpy(path, path_string.c_str(), 63);
    mkdtemp(path);

    std::string srcSuffix;
    switch (dev_.type()) {
    case TargetType::CPU:
        srcSuffix = ".cpp";
        break;
    case TargetType::GPU:
        srcSuffix = ".cu";
        break;
    default:
        ASSERT(false);
    }

    auto cpp = (std::string)path + "/run" + srcSuffix;
    auto so = (std::string)path + "/run.so";
    {
        std::ofstream f(cpp);
        f << src_;
    }
    std::string cmd;
    switch (dev_.type()) {
    case TargetType::CPU:
        cmd = "c++ -I" NAME(IR_RUNTIME_DIR) " -shared -O3 -fPIC -Wall -fopenmp";
        if (dev_.target()->useNativeArch()) {
            cmd += " -march=native";
        }
        cmd += " -o " + so + " " + cpp;
        break;
    case TargetType::GPU:
        cmd = "nvcc -I" NAME(
            IR_RUNTIME_DIR) " -shared -Xcompiler -fPIC,-Wall,-O3";
        if (auto arch = dev_.target().as<GPU>()->computeCapability();
            arch.isValid()) {
            cmd += " -arch sm_" + std::to_string(arch->first) +
                   std::to_string(arch->second);
        } else if (dev_.target()->useNativeArch()) {
            int major, minor;
            checkCudaError(cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, dev_.num()));
            checkCudaError(cudaDeviceGetAttribute(
                &minor, cudaDevAttrComputeCapabilityMinor, dev_.num()));
            cmd += " -arch sm_" + std::to_string(major) + std::to_string(minor);
        } else {
            WARNING("GPU arch not specified, which may result in suboptimal "
                    "performance ");
        }
        cmd += " -o " + so + " " + cpp;
        break;
    default:
        ASSERT(false);
    }
    system(cmd.c_str());

    dlHandle_ = dlopen(so.c_str(), RTLD_NOW);
    if (!dlHandle_) {
        throw DriverError("Unable to load target code");
    }

    func_ = (void (*)(void **))dlsym(dlHandle_, "run");
    if (!func_) {
        throw DriverError("Target function not found");
    }

    remove(cpp.c_str());
    remove(so.c_str());
    rmdir(path);
}

void Driver::setParams(const std::unordered_map<std::string, Array &> &params) {
    for (auto &&item : params) {
        params_[name2param_[item.first]] = item.second.raw();
    }
}

void Driver::run() { func_(params_.data()); }

double Driver::time(int rounds, int warmups) {
    namespace ch = std::chrono;

    double tot = 0;
    auto tgtType = dev_.type();
    for (int i = 0; i < warmups; i++) {
        run();
        switch (tgtType) {
        case TargetType::GPU:
            checkCudaError(cudaDeviceSynchronize());
        default:;
        }
    }
    for (int i = 0; i < rounds; i++) {
        auto cudaErr = cudaSuccess;

        auto beg = ch::high_resolution_clock::now();
        run();
        switch (tgtType) {
        case TargetType::GPU:
            cudaErr = cudaDeviceSynchronize();
        default:;
        }
        auto end = ch::high_resolution_clock::now();
        double dur =
            ch::duration_cast<ch::duration<double>>(end - beg).count() *
            1000; // ms

        if (cudaErr) {
            throw DriverError(cudaGetErrorString(cudaErr));
        }

        tot += dur;
    }
    return tot / rounds;
}

void Driver::unload() {
    func_ = nullptr;
    // FIXME: How to safely close it? OpenMP won't kill its worker threads
    // before it ends
    /*if (dlHandle_) {
        auto err = dlclose(dlHandle_);
        if (err) {
            WARNING("Unable to unload target code");
        }
    }*/
}

} // namespace ir
