#ifndef FREE_TENSOR_MATH_UTILS
#define FREE_TENSOR_MATH_UTILS

#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace freetensor {

// NOTE: For floating-points, we always use double to deal with compile-time
// operations

template <class T>
requires std::integral<T> T floorDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}

template <class T>
requires std::integral<T> T ceilDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}

template <class T>
requires std::integral<T> T mod(T a, T b) {
    T m = a % b;
    if (m < 0) {
        // m += (b < 0) ? -b : b; // avoid this form: it is UB when b == INT_MIN
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

template <class T>
requires std::integral<T> T gcd(T x, T y) {
    x = std::abs(x), y = std::abs(y);
    if (x < y) {
        std::swap(x, y);
    }
    do {
        T z = x % y;
        x = y;
        y = z;
    } while (y);
    return x;
}

template <class T>
requires std::integral<T> T lcm(T x, T y) { return x / gcd(x, y) * y; }

template <class T> T square(T x) { return x * x; }
inline bool square(bool x) { return x && x; }

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/**
 * Enforce casting integers to floats
 */
inline double realDiv(double a, double b) { return a / b; }

} // namespace freetensor

#endif // FREE_TENSOR_MATH_UTILS
