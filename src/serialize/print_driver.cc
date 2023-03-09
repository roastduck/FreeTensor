#include <config.h>
#include <cstring>
#include <serialize/print_driver.h>

namespace freetensor {

std::pair<std::string, std::string> dumpTarget(const Ref<Target> &target_) {

    auto target = target_.isValid() ? target_ : Config::defaultTarget();

    std::string ret_meta =
        target->toString() + " " + std::to_string(target->useNativeArch());
    std::string ret_data;

    // TODO
    switch (target->type()) {
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        auto &&tmp = target.as<GPUTarget>();
        auto deviceProp = tmp->infoArch();
        ret_data = std::string((char *)&(*deviceProp), sizeof(cudaDeviceProp));

        break;
    }
#endif // FT_WITH_CUDA
    case TargetType::CPU:
        break;

    default:
        ASSERT(false);
    }

    return std::make_pair(ret_meta, ret_data);
}

std::pair<std::string, std::string> dumpDevice(const Ref<Device> &device_) {

    auto device = device_.isValid() ? device_ : Config::defaultDevice();

    std::string ret_meta = "DEV " + std::to_string(device->num()) + " ";
    auto &&[target_meta, ret_data] = dumpTarget(device->target());
    ret_meta += target_meta;

    return std::make_pair(ret_meta, ret_data);
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
