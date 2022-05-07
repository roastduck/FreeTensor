#include <driver/target.h>

namespace freetensor {

bool isSame(const Ref<Target> &lhs, const Ref<Target> &rhs) {
    if (lhs->type() != rhs->type()) {
        return false;
    }
    if (lhs->useNativeArch() != rhs->useNativeArch()) {
        return false;
    }
    switch (lhs->type()) {
    case TargetType::CPU:
        return true;
    case TargetType::GPU: {
        auto &&l = lhs.as<GPU>(), &&r = rhs.as<GPU>();
        if (l->computeCapability() != r->computeCapability()) {
            return false;
        }
        return true;
    }
    default:
        ASSERT(false);
    }
}

} // namespace freetensor
