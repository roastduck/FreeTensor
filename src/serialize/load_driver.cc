#include <config.h>
#include <serialize/load_driver.h>

#include <cstring>
#include <iostream>
#include <sstream>

namespace freetensor {
Ref<Target> loadTarget(const std::string &txt) {

    /**
     * `CPU <useNativeArch>`
     * `GPU <useNativeArch> [: <major> <minor> | ;]`
     * e.g. `GPU 1 : 7 0`, `GPU 0 ;`, `CPU 1`
     */
    std::istringstream iss(txt);

    Ref<Target> ret;
    std::string type;
    bool useNativeArch;

    ASSERT(iss >> type >> useNativeArch);
    ASSERT(type.length() > 0);

    switch (type[0]) {
    case 'G': {
        auto ret_ = Ref<GPU>::make(useNativeArch);
        ASSERT(iss >> type);
        if (type == ":") {
            int major, minor;
            ASSERT(iss >> major >> minor);
            ret_->setComputeCapability(major, minor);
        }
        ret = ret_.as<Target>();
        break;
    }
    case 'C': {
        auto ret_ = Ref<CPU>::make(useNativeArch);
        ret = ret_.as<Target>();
        break;
    }
    default:
        ASSERT(false);
    }
    return ret;
}
Ref<Device> loadDevice(const std::string &txt) {

    /**
     * `DEV <Num> <Target>`
     * e.g. `DEV 3 GPU 1 : 7 0`
     */
    std::istringstream iss(txt);

    Ref<Device> ret;
    std::string type;
    size_t num;

    ASSERT(iss >> type >> num);
    ASSERT(type.length() > 0);

    switch (type[0]) {
    case 'D':
        // `DEV <Num> <Target>` : find a space after `<Num>`
        ret = Ref<Device>::make(loadTarget(txt.substr(txt.find(' ', 4))), num);
        break;
    default:
        ASSERT(false);
    }
    return ret;
}

Ref<Array> newArray(const std::vector<size_t> &shape_,
                    const std::string &dtypestr_,
                    const std::vector<Ref<Device>> &devs_,
                    const std::string &data_) {

    DataType dtype = parseDType(dtypestr_);
    size_t siz = sizeOf(dtype);

    for (auto len : shape_) {
        siz *= len;
    }

    // Data form: uint8_t
    ASSERT(data_.length() == siz);

    uint8_t *addr = new uint8_t[siz];
    memcpy(addr, (uint8_t *)data_.c_str(), siz);

    auto ret = Ref<Array>::make(
        Array::moveFromRaw(addr, shape_, dtype, Config::defaultDevice()));

    for (auto &dev : devs_) {
        ret->rawSharedTo(dev);
    }

    return ret;
}
Ref<Array> loadArray(const std::string &txt, const std::string &data) {

    std::istringstream iss(txt);

    Ref<Array> ret;

    std::string type;
    size_t dtype, len;
    std::vector<size_t> shape;
    std::vector<Ref<Device>> devs;

    // `ARR <dtype> <shape.size>`
    ASSERT(iss >> type >> dtype >> len);
    ASSERT(type.length() > 0);

    switch (type[0]) {
    case 'A': {

        shape.resize(len);
        for (size_t i = 0; i < len; i++) {
            ASSERT(iss >> shape[i]);
        }

        // `<ptrs_.size>`: the number of devices sharing the Array
        ASSERT(iss >> len);

        size_t st = txt.find("DEV"), ed = st;

        for (size_t i = 0; i < len; i++) {
            st = txt.find("DEV", ed);
            ed = txt.find('#', st);

            ASSERT(ed != std::string::npos);

            auto dev = loadDevice(txt.substr(st, ed - st));
            devs.emplace_back(dev);
        }

        ret = newArray(shape, dataTypeNames[dtype], devs, data);

        break;
    }

    default:
        ASSERT(false);
    }
    return ret;
}

} // namespace freetensor
