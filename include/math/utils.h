#ifndef FREE_TENSOR_MATH_UTILS
#define FREE_TENSOR_MATH_UTILS

#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace freetensor {

template <typename T>
concept IntegralExceptBool = requires {
                                 requires std::integral<T>;
                                 requires !std::same_as<T, bool>;
                             };

// NOTE: For floating-points, we always use double to deal with compile-time
// operations

inline auto floorDiv(IntegralExceptBool auto a, IntegralExceptBool auto b) {
    auto res = a / b;
    auto rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}

inline auto ceilDiv(IntegralExceptBool auto a, IntegralExceptBool auto b) {
    auto res = a / b;
    auto rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}

inline auto mod(IntegralExceptBool auto a, IntegralExceptBool auto b) {
    auto m = a % b;
    if (m < 0) {
        // m += (b < 0) ? -b : b; // avoid this form: it is UB when b == INT_MIN
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

template <IntegralExceptBool T, IntegralExceptBool U> auto gcd(T _x, U _y) {
    std::common_type_t<T, U> x = std::abs(_x), y = std::abs(_y);
    if (x < y) {
        std::swap(x, y);
    }
    do {
        auto z = x % y;
        x = y;
        y = z;
    } while (y);
    return x;
}

inline auto lcm(IntegralExceptBool auto x, IntegralExceptBool auto y) {
    return x / gcd(x, y) * y;
}

template <class T> T square(T x) { return x * x; }
inline bool square(bool x) { return x && x; }

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/**
 * Enforce casting integers to floats
 */
inline double realDiv(double a, double b) { return a / b; }

} // namespace freetensor

#endif // FREE_TENSOR_MATH_UTILS
