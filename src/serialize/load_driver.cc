#include <config.h>
#include <nlohmann/json.hpp>
#include <serialize/load_driver.h>

#include <cstring>
#include <iostream>
#include <sstream>

namespace freetensor {
std::string hexToBytes(const std::string &hex) {
    if (hex.size() % 2 != 0) {
        ERROR("Hex byte string must contain an even number of digits");
    }
    auto hexVal = [](char c) -> unsigned char {
        if ('0' <= c && c <= '9') {
            return c - '0';
        }
        if ('a' <= c && c <= 'f') {
            return c - 'a' + 10;
        }
        if ('A' <= c && c <= 'F') {
            return c - 'A' + 10;
        }
        ERROR("Invalid hexadecimal digit");
    };
    std::string ret;
    ret.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        ret.push_back((char)((hexVal(hex[i]) << 4) | hexVal(hex[i + 1])));
    }
    return ret;
}

Ref<Target> loadTarget(const std::string &txt) {

    auto j = nlohmann::json::parse(txt);
    auto type = tolower(j.at("type").get<std::string>());
    if (type == "cpu") {
        return Ref<CPUTarget>::make(j.value("use_native_arch", true));
    }
#ifdef FT_WITH_CUDA
    if (type == "gpu") {
        auto deviceProp = Ref<cudaDeviceProp>::make();
        auto bytes = hexToBytes(j.at("cuda_device_prop").get<std::string>());
        ASSERT(bytes.size() == sizeof(cudaDeviceProp));
        memcpy(&(*deviceProp), bytes.data(), sizeof(cudaDeviceProp));
        return Ref<GPUTarget>::make(deviceProp);
    }
#endif // FT_WITH_CUDA
    ERROR("Unrecognized target type " + type);
}

Ref<Device> loadDevice(const std::string &txt, const std::string &data) {

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
        // `DEV <Num> <Target JSON>` : find a space after `<Num>`
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

Ref<Array> loadArray(const std::string &txt, const std::string &data) {

    std::istringstream iss(txt);

    Ref<Array> ret;

    std::string type, dtype;
    size_t len;
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

        ret = newArray(shape, dtype, data);

        break;
    }

    default:
        ASSERT(false);
    }
    return ret;
}

} // namespace freetensor
