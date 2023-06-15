#include <omp.h>

#include <cstring>
#include <driver/target.h>

namespace freetensor {

int CPUTarget::nCores() const {
    if (useNativeArch_) {
        return omp_get_max_threads();
    } else {
        ASSERT(false); // Unimplemented
    }
}

bool isSameTarget(const Ref<Target> &lhs, const Ref<Target> &rhs) {
    if (lhs.isValid() != rhs.isValid()) {
        return false;
    }
    if (!lhs.isValid() && !rhs.isValid()) {
        return true;
    }
    if (lhs->type() != rhs->type()) {
        return false;
    }
    switch (lhs->type()) {
    case TargetType::CPU: {
        auto &&l = lhs.as<CPUTarget>(), &&r = rhs.as<CPUTarget>();
        if (l->useNativeArch() != r->useNativeArch())
            return false;
        return true;
    }
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        auto &&l = lhs.as<GPUTarget>(), &&r = rhs.as<GPUTarget>();
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
