#include <chrono>
#include <cstdio>  // remove
#include <cstdlib> // mkdtemp, system
#include <cstring> // memset
#include <dlfcn.h> // dlopen
#include <fstream>
#include <sys/stat.h> // mkdir
#include <unistd.h>   // rmdir

#include <debug.h>
#include <driver.h>
#include <driver/gpu.h>
#include <except.h>

#define NAME_(macro) #macro
#define NAME(macro) NAME_(macro)

namespace ir {

Driver::Driver(const Func &f, const std::string &src, const Device &dev)
    : f_(f), src_(src), params_(f->params_.size(), nullptr),
      returns_(f->returns_.size(), nullptr), retSizes_(f->returns_.size(), 0),
      dev_(dev) {
    auto nParams = f->params_.size();
    name2param_.reserve(nParams);
    for (size_t i = 0; i < nParams; i++) {
        name2param_[f->params_[i]] = i;
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
    auto mkdtempPtr = mkdtemp(path);
    ASSERT(mkdtempPtr != nullptr);

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
    // We enable fast-math because our own transformations do not preserve
    // strict floating point rounding order either
    switch (dev_.type()) {
    case TargetType::CPU:
        cmd = "c++ -I" NAME(
            IR_RUNTIME_DIR) " -shared -O3 -fPIC -Wall -fopenmp -ffast-math";
        cmd += " -o " + so + " " + cpp;
#ifdef WITH_MKL
        cmd += " -I\"" NAME(WITH_MKL) "/include\"";
        cmd += " -Wl,--start-group";
        cmd += " \"" NAME(WITH_MKL) "/lib/intel64/libmkl_intel_lp64.a\"";
        cmd += " \"" NAME(WITH_MKL) "/lib/intel64/libmkl_gnu_thread.a\"";
        cmd += " \"" NAME(WITH_MKL) "/lib/intel64/libmkl_core.a\"";
        cmd += " -Wl,--end-group";
        cmd += " -DWITH_MKL=\"" NAME(WITH_MKL) "\"";
        // Link statically, or there will be dlopen issues
        // Generated with MKL Link Line Advisor
#endif
        if (dev_.target()->useNativeArch()) {
            cmd += " -march=native";
        }
        break;
    case TargetType::GPU:
        cmd = "nvcc -I" NAME(IR_RUNTIME_DIR) " -shared -Xcompiler "
                                             "-fPIC,-Wall,-O3 --use_fast_math";
        cmd += " -o " + so + " " + cpp;
        cmd += " -lcublas";
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
        break;
    default:
        ASSERT(false);
    }
    auto compilerErr = system(cmd.c_str());
    if (compilerErr != 0) {
        throw DriverError("Backend compiler reports error");
    }

    dlHandle_ = dlopen(so.c_str(), RTLD_NOW);
    if (!dlHandle_) {
        throw DriverError((std::string) "Unable to load target code: " +
                          dlerror());
    }

    func_ =
        (void (*)(void **, void **, size_t *, void *))dlsym(dlHandle_, "run");
    if (!func_) {
        throw DriverError((std::string) "Target function not found: " +
                          dlerror());
    }

    remove(cpp.c_str());
    remove(so.c_str());
    rmdir(path);

    switch (dev_.type()) {
    case TargetType::CPU:
        curCtx_ = &cpuCtx_;
        break;
    case TargetType::GPU:
        curCtx_ = &gpuCtx_;
        break;
    default:
        ASSERT(false);
    }
}

void Driver::setParams(const std::vector<Ref<Array>> &args,
                       const std::unordered_map<std::string, Ref<Array>> &kws) {
    for (size_t i = 0, iEnd = args.size(), j = 0; i < iEnd; i++) {
        while (j < params_.size() && f_->closure_.count(f_->params_[j])) {
            j++;
        }
        if (j >= params_.size()) {
            throw DriverError("More arguments are given than required");
        }
        params_[j++] = args[i]->raw();
    }
    for (auto &&[key, value] : kws) {
        if (f_->closure_.count(key)) {
            throw DriverError("Enclosed parameter " + key + " cannot be set");
        }
        params_[name2param_[key]] = value->raw();
    }
    for (size_t i = 0, iEnd = params_.size(); i < iEnd; i++) {
        if (f_->closure_.count(f_->params_[i])) {
            params_[i] = (*f_->closure_.at(f_->params_[i]))->raw();
        }
        if (params_[i] == nullptr) {
            throw DriverError("Parameter " + std::to_string(i) + " is missing");
        }
    }
}

void Driver::run() {
    func_(params_.data(), returns_.data(), retSizes_.data(), curCtx_);
}

void Driver::sync() {
    switch (dev_.type()) {
    case TargetType::GPU:
        checkCudaError(cudaDeviceSynchronize());
    default:;
    }
}

std::vector<Ref<Array>> Driver::collectReturns() {
    std::vector<Ref<Array>> ret;
    for (size_t i = 0, n = f_->returns_.size(); i < n; i++) {
        auto val = Ref<Array>::make(
            Array(returns_[i], retSizes_[i], f_->returns_[i].second, dev_));
        if (f_->closure_.count(f_->returns_[i].first)) {
            *f_->closure_.at(f_->returns_[i].first) = val;
        } else {
            ret.emplace_back(val);
        }
        returns_[i] = nullptr;
        retSizes_[i] = 0;
    }
    return ret;
}

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
