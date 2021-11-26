#ifndef CPU_RUNTIME_H
#define CPU_RUNTIME_H

#include <algorithm> // min, max
#include <array>     // ByValue
#include <cassert>
#include <cmath> // INFINITY, sqrt
#include <cstdint>

#include "cpu_context.h"

#ifdef WITH_MKL
#include <mkl.h>
#endif

#define restrict __restrict__
#define __ByValArray std::array

template <class T> T floorDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}
template <class T> T ceilDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}
template <class T> T runtime_mod(T a, T b) {
    T m = a % b;
    if (m < 0) {
        // m += (b < 0) ? -b : b; // avoid this form: it is UB when b == INT_MIN
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

template <class T> T runtime_square(T x) { return x * x; }

template <class T> T runtime_sigmoid(T x) { return 1.0 / (1.0 + std::exp(-x)); }

#endif // CPU_RUNTIME_H
