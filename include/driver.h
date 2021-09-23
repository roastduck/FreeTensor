#ifndef DRIVER_H
#define DRIVER_H

#include <string>
#include <unordered_map>
#include <vector>

#include <driver/array.h>
#include <func.h>

#include <../runtime/cpu_context.h>
#include <../runtime/gpu_context.h>

namespace ir {

class Driver {
    void *dlHandle_ = nullptr;
    void (*func_)(void **, void *) = nullptr;

    std::string src_;
    std::vector<void *> params_;
    std::unordered_map<std::string, size_t> name2param_;
    Device dev_;

    CPUContext cpuCtx_;
    GPUContext gpuCtx_;
    void *curCtx_ = nullptr;

  private:
    void buildAndLoad();

  public:
    Driver(const Func &func, const std::string &src, const Device &dev);
    ~Driver() { unload(); }

    void setParams(const std::vector<Array *> &args,
                   const std::unordered_map<std::string, Array *> &kws = {});
    void setParams(const std::unordered_map<std::string, Array *> &kws) {
        setParams({}, kws);
    }

    void run();

    /**
     * Sync with the device
     *
     * Call this if you are timing without the `time` function
     */
    void sync();

    /**
     * Run the program and measure its time cost
     *
     * @param rounds : Run this amount of rounds, and report the average
     * @param warmups : Run this amount of rounds before actual measurement
     * @return : The time, in ms
     */
    double time(int rounds = 10, int warmups = 3);

    void unload();
};

} // namespace ir

#endif // DRIVER_H
