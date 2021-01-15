#ifndef DRIVER_H
#define DRIVER_H

#include <string>
#include <vector>

#include <driver/array.h>

namespace ir {

class Driver {
    void *dlHandle_ = nullptr;
    void (*func_)(void **) = nullptr;
    void **params_ = nullptr;

  public:
    ~Driver() { unload(); }
    void buildAndLoad(const std::string &src, int nParam);
    void setParam(int nth, const Array &param) { params_[nth] = param.raw(); }
    void run();
    void unload();
};

} // namespace ir

#endif // DRIVER_H
