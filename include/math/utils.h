#ifndef MATH_UTILS
#define MATH_UTILS

#include <cstdlib>
#include <utility>

namespace ir {

template <class T> T floorDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}

template <class T> T ceilDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}

template <class T> T mod(T a, T b) {
    T m = a % b;
    if (m < 0) {
        // m += (b < 0) ? -b : b; // avoid this form: it is UB when b == INT_MIN
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

template <class T> T gcd(T x, T y) {
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

template <class T> T lcm(T x, T y) { return x / gcd(x, y) * y; }

} // namespace ir

#endif // MATH_UTILS
