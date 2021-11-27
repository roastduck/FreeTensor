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
    void (*func_)(void ** /* params */, void ** /* retRaw */,
                  size_t ** /* retShapes */, size_t * /* retDims */,
                  void * /* ctx */) = nullptr;

    Func f_;
    std::string src_;
    std::vector<void *> params_, returns_;
    std::vector<size_t *> retShapes_;
    std::vector<size_t> retDims_;
    std::unordered_map<std::string, size_t> name2param_;
    Device dev_;

    Context *ctx_ = nullptr;

  private:
    void buildAndLoad();

  public:
    Driver(const Func &func, const std::string &src, const Device &dev);
    ~Driver() {
        for (void *retVal : returns_) {
            if (retVal != nullptr) {
                WARNING("Return values must be collected, or there will be "
                        "memory leaks");
            }
        }
        unload();
        if (ctx_ != nullptr) {
            delete ctx_;
        }
    }

    Driver(const Driver &) = delete;
    Driver &operator=(const Driver &) = delete;

    Driver(Driver &&) = default;
    Driver &operator=(Driver &&) = default;

    void setParams(const std::vector<Ref<Array>> &args,
                   const std::unordered_map<std::string, Ref<Array>> &kws = {});
    void setParams(const std::unordered_map<std::string, Ref<Array>> &kws) {
        setParams({}, kws);
    }

    void run();

    /**
     * Sync with the device
     *
     * Call this if you are timing without the `time` function
     */
    void sync();

    std::vector<Ref<Array>> collectReturns();

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
