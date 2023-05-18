#ifndef FREETENSOR_TIMEOUT_H
#define FREETENSOR_TIMEOUT_H

#include <functional>
#include <vector>

namespace freetensor {

/**
 * Run a function with timeout
 *
 * Fork a new process to run. If success, return a byte-vector result. If
 * timeout, return an empty vector
 */
std::vector<std::byte>
timeout(const std::function<std::vector<std::byte>()> &func, int seconds);

} // namespace freetensor

#endif // FREETENSOR_TIMEOUT_H
