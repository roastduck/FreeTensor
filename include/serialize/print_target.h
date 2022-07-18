#ifndef FREE_TENSOR_PRINT_TARGET_H
#define FREE_TENSOR_PRINT_TARGET_H

// for multi-machine-parallel xmlrpc

#include <string>

#include <ref.h>
#include <driver/target.h>

namespace freetensor{

inline std::string dumpTarget(const Ref<Target> &target) {

    std::string ret = target->toString() + " " + std::to_string(target->useNativeArch());

    switch (target->type()) {
    case TargetType::GPU:
        {
            Ref<GPU> _target = target.as<GPU>();
            Opt<std::pair<int, int>> _computeCapability = _target->computeCapability();
            if (_computeCapability.isValid()) {
                ret += " : " + std::to_string(_computeCapability->first) + " " + std::to_string(_computeCapability->second);
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

} // namespace freetensor

#endif // FREE_TENSOR_PRINT_TARGET_H
