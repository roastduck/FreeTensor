#include <config.h>
#include <cstring>
#include <serialize/print_driver.h>

namespace freetensor {

std::string dumpTarget(const Ref<Target> &target_) {

    auto target = target_.isValid() ? target_ : Config::defaultTarget();

    std::string ret =
        target->toString() + " " + std::to_string(target->useNativeArch());

    switch (target->type()) {
    case TargetType::GPU: {
        auto _target = target.as<GPU>();
        auto _computeCapability = _target->computeCapability();
        if (_computeCapability.isValid()) {
            ret += " : " + std::to_string(_computeCapability->first) + " " +
                   std::to_string(_computeCapability->second);
        } else {
            ret += " ;";
        }
        break;
    }
    case TargetType::CPU:
        break;

    default:
        ASSERT(false);
    }

    return ret;
}

std::string dumpDevice(const Ref<Device> &device_) {

    auto device = device_.isValid() ? device_ : Config::defaultDevice();

    std::string ret = "DEV " + std::to_string(device->num()) + " " +
                      dumpTarget(device->target());
    return ret;
}

std::string dumpArray(const Ref<Array> &array_) {

    /**
     * The string is constructed as follow (Separated by space):
     *
     * {"ARR"} +
     * {dtype} +
     * {shape.size} + {shape[0]} + ... + {shape[shape.size - 1]} +
     * {ptrs.size} + {ptrs[0].dev_"#"} + ... + {ptrs[ptrs.size - 1].dev_"#"} +
     * {Arraydata (uint8_t, strlen = array_->size(), e.g.!@#$%^&*1)}
     *
     * Note: Arraydata may have invisible characters
     */
    ASSERT(array_.isValid());

    auto array = array_;

    // it will modify array_
    uint8_t *addr = (uint8_t *)array->rawSharedTo(Config::defaultDevice());

    ASSERT(addr);

    std::string ret = "ARR " + std::to_string((size_t)array->dtype()) + " " +
                      std::to_string(array->shape().size()) + " ";

    for (const size_t &siz : array->shape()) {
        ret += std::to_string(siz) + " ";
    }

    ret += std::to_string(array_->ptrs().size()) + " ";
    // `array_` is correct
    for (auto &&[device, p, _] : array_->ptrs()) {
        ret += dumpDevice(device) + "# ";
    }

    for (size_t i = 0; i < array->size(); i++) {
        ret += addr[i];
    }

    return ret;
}

} // namespace freetensor
