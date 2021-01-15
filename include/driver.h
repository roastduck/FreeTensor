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

  private:
    void buildAndLoad();

  public:
    Driver(const std::string &src, const std::vector<std::string> &paramNames);
    ~Driver() { unload(); }

    void setParams(const std::unordered_map<std::string, Array &> &params);
    void run();
    void unload();
};

} // namespace ir

#endif // DRIVER_H
