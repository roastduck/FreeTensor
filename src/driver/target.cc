#include <driver/target.h>
#include <cstring>

namespace freetensor {
bool isSameTarget(const Ref<Target> &lhs, const Ref<Target> &rhs) {
    if (lhs->type() != rhs->type()) {
        return false;
    }
    if (lhs->useNativeArch() != rhs->useNativeArch()) {
        return false;
    }
    switch (lhs->type()) {
    case TargetType::CPU:

        return true;
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        auto &&l = lhs.as<GPU>(), &&r = rhs.as<GPU>();
        uint8_t *lptr = (uint8_t *)&(*(l->infoArch()));
        uint8_t *rptr = (uint8_t *)&(*(r->infoArch()));
        if (memcmp(lptr, rptr, sizeof(cudaDeviceProp)) == 0)
            return true;
        return false;
    }
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
}

} // namespace freetensor
