#include <cstdio>  // remove
#include <cstdlib> // mkdtemp, system
#include <cstring> // memset
#include <dlfcn.h> // dlopen
#include <fstream>
#include <sys/stat.h> // mkdir
#include <unistd.h>   // rmdir

#include "driver.h"
#include "except.h"

namespace ir {

void Driver::buildAndLoad(const std::string &src, int nParam) {
    mkdir("/tmp/ir", 0755);
    char path[] = "/tmp/ir/XXXXXX";
    mkdtemp(path);

    auto cpp = (std::string)path + "/run.cpp";
    auto so = (std::string)path + "/run.so";
    {
        std::ofstream f(cpp);
        f << src;
    }
    auto cmd =
        (std::string) "c++ -shared -fPIC -Wall -fopenmp -o " + so + " " + cpp;
    system(cmd.c_str());

    dlHandle_ = dlopen(so.c_str(), RTLD_NOW);
    if (!dlHandle_) {
        ERROR("Unable to load target code");
    }

    func_ = (void (*)(void **))dlsym(dlHandle_, "run");
    if (!func_) {
        ERROR("Target function not found");
    }

    params_ = new void *[nParam];
    memset(params_, 0, nParam * sizeof(void *));

    remove(cpp.c_str());
    remove(so.c_str());
    rmdir(path);
}

void Driver::run() { func_(params_); }

void Driver::unload() {
    delete[] params_;
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

