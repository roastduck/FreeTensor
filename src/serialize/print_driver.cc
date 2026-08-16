#include <config.h>
#include <cstring>
#include <nlohmann/json.hpp>
#include <serialize/print_driver.h>

namespace freetensor {

std::string bytesToHex(const void *data, size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    auto bytes = reinterpret_cast<const unsigned char *>(data);
    std::string ret;
    ret.reserve(size * 2);
    for (size_t i = 0; i < size; i++) {
        ret.push_back(digits[bytes[i] >> 4]);
        ret.push_back(digits[bytes[i] & 0xf]);
    }
    return ret;
}

std::string dumpTarget(const Ref<Target> &target_) {

    auto target = target_.isValid() ? target_ : Config::defaultTarget();

    nlohmann::json ret;
    ret["type"] = target->type() == TargetType::CPU ? "cpu" : "gpu";
    ret["use_native_arch"] = target->useNativeArch();

    switch (target->type()) {
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        auto &&tmp = target.as<GPUTarget>();
        auto deviceProp = tmp->infoArch();
        ret["cuda_device_prop"] =
            bytesToHex(&(*deviceProp), sizeof(cudaDeviceProp));
        break;
    }
#endif // FT_WITH_CUDA
    case TargetType::CPU:
        break;

    default:
        ASSERT(false);
    }

    return ret.dump();
}

std::pair<std::string, std::string> dumpDevice(const Ref<Device> &device_) {

    auto device = device_.isValid() ? device_ : Config::defaultDevice();

    std::string ret_meta = "DEV " + std::to_string(device->num()) + " ";
    ret_meta += dumpTarget(device->target());

    return std::make_pair(ret_meta, std::string{});
}

std::pair<std::string, std::string> dumpArray(const Ref<Array> &array_) {

    /**
     * The string is constructed as follow (Separated by space):
     *
     * MetaData:
     * {"ARR"} +
     * {dtype} +
     * {shape.size} + {shape[0]} + ... + {shape[shape.size - 1]}
     *
     * Data:
     * {Arraydata (string -> pybind11::bytes later, e.g. b'\x01\x23\xab\xcd')
     *
     *
     */
    ASSERT(array_.isValid());

    auto array = array_;

    // array_ may be modified
    uint8_t *addr = (uint8_t *)array->rawSharedTo(Config::defaultDevice());

    ASSERT(addr);

    std::string ret_meta = "ARR " + toString(array->dtype()) + " " +
                           std::to_string(array->shape().size()) + " ";

    for (const size_t &siz : array->shape()) {
        ret_meta += std::to_string(siz) + " ";
    }

    std::string ret_data((char *)addr, array->size());

    return std::make_pair(ret_meta, ret_data);
}

} // namespace freetensor
