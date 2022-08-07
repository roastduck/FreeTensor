#include <config.h>
#include <serialize/load_driver.h>

#include <cstring>
#include <iostream>
#include <sstream>

namespace freetensor {
Ref<Target> loadTarget(const std::string &txt) {

    /**
     * `[CPU|GPU] <useNativeArch>`
     */
    std::istringstream iss(txt);

    Ref<Target> ret;
    std::string type;
    bool useNativeArch;

    ASSERT(iss >> type >> useNativeArch);
    ASSERT(type.length() > 0);

    switch (type[0]) {
#ifndef FT_WITH_CUDA
    case 'G': {

        auto ret_ = Ref<GPU>::make(useNativeArch);
        ret = ret_.as<Target>();
        break;
    }

#endif // NOT FT_WITH_CUDA
#ifdef FT_WITH_CUDA
    case 'G': {
        auto ret_ = Ref<GPU>::make(nullptr, useNativeArch);
        // TODO
        ret = ret_.as<Target>();
        break;
    }
#endif // FT_WITH_CUDA
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
     * e.g. `DEV 3 GPU 1`
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
        ret = Ref<Device>::make(
            loadTarget(txt.substr(txt.find(' ', 4)))->type(), num);
        break;
    default:
        ASSERT(false);
    }
    return ret;
}

Ref<Array> newArray(const std::vector<size_t> &shape_,
                    const std::string &dtypestr_, const std::string &data_) {

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

    return ret;
}
Ref<Array>
loadArray(const std::pair<const std::string &, const std::string &> &txt_data) {

    auto &&[txt, data] = txt_data;
    std::istringstream iss(txt);

    Ref<Array> ret;

    std::string type;
    size_t dtype, len;
    std::vector<size_t> shape;

    // `ARR <dtype> <shape.size>`
    ASSERT(iss >> type >> dtype >> len);
    ASSERT(type.length() > 0);

    switch (type[0]) {
    case 'A': {

        shape.resize(len);
        for (size_t i = 0; i < len; i++) {
            ASSERT(iss >> shape[i]);
        }

        ret = newArray(shape, dataTypeNames[dtype], data);

        break;
    }

    default:
        ASSERT(false);
    }
    return ret;
}

} // namespace freetensor
