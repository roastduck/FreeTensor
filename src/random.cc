#include <omp.h>

#include <random.h>

namespace freetensor {

OpenMPRandomEngine::OpenMPRandomEngine(RNG::result_type seed) {
    RNG rng0(seed);
    int nthr = omp_get_max_threads();
    rngs_.reserve(nthr);
    for (int i = 0; i < nthr; i++) {
        rngs_.emplace_back(rng0());
    }
}

OpenMPRandomEngine::RNG::result_type OpenMPRandomEngine::operator()() {
    return rngs_.at(omp_get_thread_num())();
}

} // namespace freetensor
