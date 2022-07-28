#include <chrono>
#include <cmath>   // sqrt
#include <cstdio>  // remove
#include <cstdlib> // mkdtemp, system
#include <cstring> // memset
#include <dlfcn.h> // dlopen
#include <fstream>
#include <sys/stat.h> // mkdir
#include <sys/wait.h> // waitpid
#include <unistd.h>   // rmdir

#include <itertools.hpp>

#include <analyze/find_stmt.h>
#include <config.h>
#include <debug.h>
#include <driver.h>
#include <except.h>
#ifdef FT_WITH_CUDA
#include <driver/gpu.h>
#endif

namespace freetensor {

static void *requestPtr(const Ref<Array> &arr, const Ref<Device> &device,
                        const Ref<Device> &hostDevice, MemType mtype,
                        AccessType atype) {
    Ref<Device> d;
    switch (mtype) {
    case MemType::CPU:
    case MemType::ByValue:
        if (hostDevice->type() != TargetType::CPU) {
            throw DriverError("A CPU host device is requested");
        }
        d = hostDevice;
        break;
    case MemType::GPUGlobal:
        if (device->type() != TargetType::GPU) {
            throw DriverError("A GPU device is requested");
        }
        d = device;
        break;
    default:
        throw DriverError("A I/O variable cannot have a " + toString(mtype) +
                          " memory type");
    }
    switch (atype) {
    case AccessType::Input:
        return arr->rawSharedTo(d);
    case AccessType::Output:
        return arr->rawInitTo(d);
    case AccessType::InOut:
        return arr->rawMovedTo(d);
    case AccessType::Cache:
        throw DriverError("A \"cache\" variable cannot be an I/O variable");
    default:
        ASSERT(false);
    }
}

Driver::Driver(const Func &f, const std::string &src, const Ref<Device> &dev,
               const Ref<Device> &hostDev, bool verbose)
    : f_(f), src_(src), args_(f->params_.size(), nullptr),
      rawArgs(f->params_.size(), nullptr), rawRets(f->returns_.size(), nullptr),
      retShapes_(f->returns_.size(), nullptr), retDims_(f->returns_.size(), 0),
      dev_(dev), hostDev_(hostDev), verbose_(verbose) {
    auto nParams = f->params_.size();
    name2param_.reserve(nParams);
    name2buffer_.reserve(nParams);
    for (size_t i = 0; i < nParams; i++) {
        name2param_[f->params_[i].name_] = i;
        auto nodes = findStmt(f->body_, [&](const Stmt &s) -> bool {
            return s->nodeType() == ASTNodeType::VarDef &&
                   s.as<VarDefNode>()->name_ == f->params_[i].name_;
        });
        if (nodes.size() != 1) {
            throw DriverError(
                "Name " + f->params_[i].name_ +
                " should be existent and unique in the AST as a paramerter");
        }
        name2buffer_[f->params_[i].name_] =
            nodes.front().as<VarDefNode>()->buffer_;
    }
    buildAndLoad();
}

void Driver::buildAndLoad() {
    std::string home = getenv("HOME");
    mkdir((home + "/.freetensor").c_str(), 0755);
    std::string path_string = home + "/.freetensor/XXXXXX";
    char path[64];
    ASSERT(path_string.size() < 64);
    strncpy(path, path_string.c_str(), 63);
    auto mkdtempPtr = mkdtemp(path);
    ASSERT(mkdtempPtr != nullptr);

    std::string srcSuffix;
    switch (dev_->type()) {
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
    const char *executable;
    std::vector<std::string> args;
    auto addArgs = [&](auto... s) {
        args.insert(args.end(), {std::string(s)...});
    };
    // We enable fast-math because our own transformations do not preserve
    // strict floating point rounding order either
    switch (dev_->type()) {
    case TargetType::CPU:
        executable = Config::backendCompilerCXX().c_str();
        // For path arguments, we do not quote it again since the arguments are passed
        // directly to the compiler (with execv) without going through the shell. Spaces
        // are preserved and the argument will not be split into multiple arguments.
        addArgs("-I" FT_RUNTIME_DIR, "-std=c++20", "-shared", "-O3", "-fPIC",
                "-Wall", "-fopenmp", "-ffast-math");
        addArgs("-o", so, cpp);
#ifdef FT_WITH_MKL
        addArgs("-I" FT_WITH_MKL "/include", "-Wl,--start-group",
                FT_WITH_MKL "/lib/intel64/libmkl_intel_lp64.a",
                FT_WITH_MKL "/lib/intel64/libmkl_gnu_thread.a",
                FT_WITH_MKL "/lib/intel64/libmkl_core.a", "-Wl,--end-group",
                "-DFT_WITH_MKL=" FT_WITH_MKL);
        // Link statically, or there will be dlopen issues
        // Generated with MKL Link Line Advisor
#endif // FT_WITH_MKL
        if (dev_->target()->useNativeArch()) {
            addArgs("-march=native");
        }
        if (Config::debugBinary()) {
            addArgs("-g");
        }
        break;
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        executable = Config::backendCompilerNVCC().c_str();
        addArgs("-I" FT_RUNTIME_DIR, "-std=c++17", "-shared", "-Xcompiler",
                "-fPIC,-Wall,-O3", "--use_fast_math");
        addArgs("-o", so, cpp);
        addArgs("-lcublas");
        if (auto arch = dev_->target().as<GPU>()->computeCapability();
            arch.isValid()) {
            addArgs("-arch", "sm_" + std::to_string(arch->first) +
                                 std::to_string(arch->second));
        } else if (dev_->target()->useNativeArch()) {
            int major, minor;
            checkCudaError(cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, dev_->num()));
            checkCudaError(cudaDeviceGetAttribute(
                &minor, cudaDevAttrComputeCapabilityMinor, dev_->num()));
            addArgs("-arch",
                    "sm_" + std::to_string(major) + std::to_string(minor));
        } else {
            WARNING("GPU arch not specified, which may result in suboptimal "
                    "performance ");
        }
        if (Config::debugBinary()) {
            addArgs("-g");
        }
        break;
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }

    if (Config::debugBinary() || verbose_) {
        std::stringstream cmdStream;
        cmdStream << "\"" << executable << "\" ";
        for (auto &s : args) {
            cmdStream << "\"" << s << "\" ";
        }
        auto cmd = cmdStream.str();

        if (Config::debugBinary()) {
            WARNING("debug-binary mode on. Compiling with " + cmd);
        }
        if (verbose_) {
            logger() << "Running " << cmd << std::endl;
        }
    }

    // fork + execv to execute the compiler
    {
        // construct the argv array
        std::vector<const char *> argv;
        argv.push_back(executable);
        for (auto &s : args) {
            argv.push_back(s.c_str());
        }
        argv.push_back(nullptr);

        int pid = fork();
        if (pid == 0) {
            execv(executable, const_cast<char *const *>(argv.data()));
            std::cerr << "Failed to execute " << executable << ": "
                      << strerror(errno);
            exit(-1);
        } else {
            int status;
            waitpid(pid, &status, 0);
            if (status != 0)
                throw DriverError("Backend compiler reports error");
        }
    }

    dlHandle_ = dlopen(so.c_str(), RTLD_NOW);
    if (!dlHandle_) {
        throw DriverError((std::string) "Unable to load target code: " +
                          dlerror());
    }

    func_ = (void (*)(void **, void **, size_t **, size_t *, void *))dlsym(
        dlHandle_, "run");
    if (!func_) {
        throw DriverError((std::string) "Target function not found: " +
                          dlerror());
    }

    if (!Config::debugBinary()) {
        remove(cpp.c_str());
        remove(so.c_str());
        rmdir(path);
    } else {
        WARNING((std::string) "debug-binary mode on. The produced files are "
                              "saved in " +
                path);
    }

    switch (dev_->type()) {
    case TargetType::CPU:
        ctx_ = std::make_unique<CPUContext>();
        break;
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        ctx_ = std::make_unique<GPUContext>();
        break;
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
}

void Driver::setArgs(const std::vector<Ref<Array>> &args,
                     const std::unordered_map<std::string, Ref<Array>> &kws) {
    for (size_t i = 0, iEnd = args.size(), j = 0; i < iEnd; i++) {
        while (j < rawArgs.size() && f_->params_[j].isInClosure() &&
               !f_->params_[j].updateClosure_) {
            j++;
        }
        if (j >= rawArgs.size()) {
            throw DriverError("More arguments are given than required");
        }
        auto &&buffer = name2buffer_.at(f_->params_[j].name_);
        if (buffer->tensor()->dtype() != args[i]->dtype()) {
            throw DriverError(
                "Cannnot pass a " + toString(args[i]->dtype()) +
                " Array to the " + std::to_string(j) + "-th parameter " +
                f_->params_[j].name_ + " of type " +
                toString(
                    name2buffer_.at(f_->params_[j].name_)->tensor()->dtype()));
        }
        args_[j] = args[i];
        rawArgs[j] = requestPtr(args[i], dev_, hostDev_, buffer->mtype(),
                                buffer->atype());
        if (f_->params_[j].isInClosure() && f_->params_[j].updateClosure_) {
            *f_->params_[j].closure_ = args[j];
        }
        j++;
    }
    for (auto &&[key, value] : kws) {
        if (!name2param_.count(key)) {
            throw DriverError("There is no parameter named " + key);
        }
        auto &&buffer = name2buffer_.at(key);
        if (buffer->tensor()->dtype() != value->dtype()) {
            throw DriverError(
                "Cannnot pass a " + toString(value->dtype()) +
                " Array to the " + std::to_string(name2param_[key]) +
                "-th parameter " + key + " of type " +
                toString(name2buffer_.at(key)->tensor()->dtype()));
        }
        auto paramId = name2param_[key];
        args_[paramId] = value;
        rawArgs[paramId] =
            requestPtr(value, dev_, hostDev_, buffer->mtype(), buffer->atype());
        if (f_->params_[paramId].isInClosure()) {
            if (f_->params_[paramId].updateClosure_) {
                *f_->params_[paramId].closure_ = value;
            } else {
                throw DriverError("Enclosed parameter " + key +
                                  " cannot be set");
            }
        }
    }
    for (auto &&[i, rawArg, param] :
         iter::zip(iter::count(), rawArgs, f_->params_)) {
        auto &&buffer = name2buffer_.at(param.name_);
        if (rawArg == nullptr && param.isInClosure()) {
            if (!param.closure_->isValid()) {
                throw DriverError("Closure variable " + param.name_ +
                                  " is not set");
            }
            rawArg = requestPtr(*param.closure_, dev_, hostDev_,
                                buffer->mtype(), buffer->atype());
        }
        if (rawArg == nullptr) {
            throw DriverError("The " + std::to_string(i) + "-th parameter " +
                              param.name_ + " is missing");
        }
    }
}

void Driver::run() {
#ifdef FT_WITH_CUDA
    if (dev_->type() == TargetType::GPU) {
        checkCudaError(cudaSetDevice(dev_->num()));
    }
#endif // FT_WITH_CUDA
    func_(rawArgs.data(), rawRets.data(), retShapes_.data(), retDims_.data(),
          ctx_.get());
}

void Driver::sync() { dev_->sync(); }

std::vector<Ref<Array>> Driver::collectReturns() {
    std::vector<Ref<Array>> ret;
    for (size_t i = 0, n = f_->returns_.size(); i < n; i++) {
        auto &&[name, dtype, closure, returnClosure] = f_->returns_[i];
        Ref<Array> val;
        if (name2param_.count(name)) {
            // Returning an argument
            val = args_.at(name2param_.at(name));
        } else {
            std::vector<size_t> shape(retShapes_[i],
                                      retShapes_[i] + retDims_[i]);
            val = Ref<Array>::make(
                Array::moveFromRaw(rawRets[i], shape, dtype, dev_));
            if (retShapes_[i] != nullptr) {
                free(retShapes_[i]);
            }
            rawRets[i] = nullptr;
            retShapes_[i] = nullptr;
            retDims_[i] = 0;
        }
        if (closure.isValid()) {
            *closure = val;
            if (returnClosure) {
                ret.emplace_back(val);
            }
        } else {
            ret.emplace_back(val);
        }
    }

    // Free reference count holders
    std::fill(args_.begin(), args_.end(), nullptr);
    std::fill(rawArgs.begin(), rawArgs.end(), nullptr);

    return ret;
}

std::pair<double, double> Driver::time(int rounds, int warmups) {
    namespace ch = std::chrono;

    std::vector<double> times(rounds);

    auto tgtType = dev_->type();
    for (int i = 0; i < warmups; i++) {
        run();
        switch (tgtType) {
#ifdef FT_WITH_CUDA
        case TargetType::GPU:
            checkCudaError(cudaDeviceSynchronize());
#endif // FT_WITH_CUDA
        default:;
        }
    }
    for (int i = 0; i < rounds; i++) {
#ifdef FT_WITH_CUDA
        auto cudaErr = cudaSuccess;
#endif // FT_WITH_CUDA

        auto beg = ch::high_resolution_clock::now();
        run();
        switch (tgtType) {
#ifdef FT_WITH_CUDA
        case TargetType::GPU:
            cudaErr = cudaDeviceSynchronize();
#endif // FT_WITH_CUDA
        default:;
        }
        auto end = ch::high_resolution_clock::now();
        double dur =
            ch::duration_cast<ch::duration<double>>(end - beg).count() *
            1000; // ms

#ifdef FT_WITH_CUDA
        if (cudaErr) {
            throw DriverError(cudaGetErrorString(cudaErr));
        }
#endif // FT_WITH_CUDA

        times[i] = dur;
    }

    double avg = 0, varAvgX = 0;
    for (auto t : times) {
        avg += t;
    }
    avg /= rounds;
    if (rounds > 1) {
        double varX = 0;
        for (auto t : times) {
            varX += (t - avg) * (t - avg);
        }
        varX /= (rounds - 1);    // Var[X] = n/(n-1) sigma^2
        varAvgX = varX / rounds; // Var[(X1 + X2 + ... + Xn) / n] = 1/n Var[X]
    }
    return std::make_pair(avg, sqrt(varAvgX));
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

} // namespace freetensor
