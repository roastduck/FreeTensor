#ifndef DRIVER_H
#define DRIVER_H

#include <string>
#include <unordered_map>
#include <vector>

#include <driver/array.h>

namespace ir {

class Driver {
    void *dlHandle_ = nullptr;
    void (*func_)(void **) = nullptr;

    std::string src_;
    std::vector<void *> params_;
    std::unordered_map<std::string, size_t> name2param_;
    Device dev_;

  private:
    void buildAndLoad();

  public:
    Driver(const std::string &src, const std::vector<std::string> &paramNames,
           const Device &dev);
    ~Driver() { unload(); }

    void setParams(const std::unordered_map<std::string, Array &> &params);
    void run();

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
