#include <hash_combine.h>

namespace freetensor {

size_t hashCombine(size_t seed, size_t other) {
    // https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values
    return seed ^ (other + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

} // namespace freetensor
