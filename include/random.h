#ifndef FREE_TENSOR_RANDOM_H
#define FREE_TENSOR_RANDOM_H

#include <random>
#include <vector>

namespace freetensor {

/**
 * Thread-safe random engine for OpenMP threads
 *
 * Each thread owns its private sub- random engine, and calls to
 * OpenMPRandomEngine is lock-free
 *
 * For debugging with deterministic psudo random numbers, just ensure two
 * things:
 *
 * 1. The seed is deterministic
 * 2. Tasks are distrubuted to threads in a deterministic way (`parallel for
 * schedule(static)`)
 */
class OpenMPRandomEngine {
    typedef std::default_random_engine RNG;

    std::vector<RNG> rngs_; /// sub-engines for each thread

  public:
    typedef RNG::result_type
        result_type; /// Requried by STL, but not enforced in
                     /// std::uniform_random_bit_generator

    OpenMPRandomEngine(RNG::result_type seed);

    /**
     * Disable copy so we won't accidentally repeat some sequences
     *
     * @{
     */
    OpenMPRandomEngine(const OpenMPRandomEngine &) = delete;
    OpenMPRandomEngine &operator=(const OpenMPRandomEngine &) = delete;
    /** @} */

    OpenMPRandomEngine(OpenMPRandomEngine &&) = default;
    OpenMPRandomEngine &operator=(OpenMPRandomEngine &&) = default;

    RNG::result_type operator()();

    static constexpr RNG::result_type min() { return RNG::min(); }
    static constexpr RNG::result_type max() { return RNG::max(); }
};

static_assert(std::uniform_random_bit_generator<OpenMPRandomEngine>);

} // namespace freetensor

#endif // FREE_TENSOR_RANDOM_H
