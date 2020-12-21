#include <cstdio>  // remove
#include <cstdlib> // mkdtemp, system
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
    {
        std::ofstream f((std::string)path + ".cpp");
        f << src;
    }
    auto cpp = (std::string)path + ".cpp";
    auto so = (std::string)path + ".so";
    auto cmd = (std::string) "c++ -shared -fPIC -o " + so + " " + cpp;
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

    remove(cpp.c_str());
    remove(so.c_str());
    rmdir(path);
}

void Driver::run() { func_(params_); }

void Driver::unload() {
    delete[] params_;
    func_ = nullptr;
    if (dlHandle_) {
        auto err = dlclose(dlHandle_);
        if (err) {
            WARNING("Unable to unload target code");
        }
    }
}

} // namespace ir

