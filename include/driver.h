#ifndef FREE_TENSOR_DRIVER_H
#define FREE_TENSOR_DRIVER_H

#include <string>
#include <unordered_map>
#include <vector>

#include <driver/array.h>
#include <func.h>

#include <../runtime/cpu_context.h>
#ifdef FT_WITH_CUDA
#include <../runtime/gpu_context.h>
#endif

namespace freetensor {

class Driver {
    void *dlHandle_ = nullptr;
    void (*func_)(void ** /* params */, void ** /* retRaw */,
                  size_t ** /* retShapes */, size_t * /* retDims */,
                  void * /* ctx */) = nullptr;

    Func f_;
    std::string src_;
    std::vector<Ref<Array>> args_; /// Ref count holders
    std::vector<void *> rawArgs,
        rawRets; /// Raw arguments and return values passed to (from) the
                 /// native function
    std::vector<size_t *> retShapes_;
    std::vector<size_t> retDims_;
    std::unordered_map<std::string, size_t> name2param_;
    std::unordered_map<std::string, Ref<Buffer>> name2buffer_;
    Ref<Device> dev_, hostDev_;

    std::unique_ptr<Context> ctx_;

    bool verbose_ = false;

  private:
    void buildAndLoad();

  public:
    /**
     * Compile a program using a backend compiler and load it into memory
     *
     * @param func : AST of the function, where the function signature is needed
     * to determine the parameters and return values
     * @param src : Native code generated from codegen
     * @param device : The device to run the program
     * @param hostDevice : The hosting CPU device (Optional)
     * @{
     */
    Driver(const Func &func, const std::string &src, const Ref<Device> &device,
           const Ref<Device> &hostDevice, bool verbose = false);
    Driver(const Func &func, const std::string &src, const Ref<Device> &device,
           bool verbose = false)
        : Driver(func, src, device,
                 device->type() == TargetType::CPU
                     ? device
                     : Ref<Device>::make(Ref<CPU>::make()),
                 verbose) {}
    /** @} */

    ~Driver() {
        for (void *retVal : rawRets) {
            if (retVal != nullptr) {
                WARNING("Return values must be collected, or there will be "
                        "memory leaks");
            }
        }
        unload();
    }

    Driver(const Driver &) = delete;
    Driver &operator=(const Driver &) = delete;

    Driver(Driver &&) = default;
    Driver &operator=(Driver &&) = default;

    void setArgs(const std::vector<Ref<Array>> &args,
                 const std::unordered_map<std::string, Ref<Array>> &kws = {});
    void setArgs(const std::unordered_map<std::string, Ref<Array>> &kws) {
        setArgs({}, kws);
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
     * @return : (average time, estimated standard deviation of the average
     * time = sqrt(Var(X1 + X2 + ... + Xn))), in ms
     */
    std::pair<double, double> time(int rounds = 10, int warmups = 3);

    void unload();
};

} // namespace freetensor

#endif // FREE_TENSOR_DRIVER_H
