#ifndef FREE_TENSOR_STREAM_UTILS_H
#define FREE_TENSOR_STREAM_UTILS_H

#include <iostream>

namespace freetensor {

/**
 * Tell std::ostream not only to treat the 3 function types in the standard as
 * manipulators, but also arbitrary invocable types
 */
template <typename Func>
auto operator<<(std::ostream &os, const Func &func) -> decltype(func(os)) {
    return func(os);
}

} // namespace freetensor

#endif // FREE_TENSOR_STREAM_UTILS_H
