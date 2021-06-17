#ifndef IR_UTILS_H
#define IR_UTILS_H

#include <array>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>

namespace ir {

template<int n>
std::array<int, n> random_fill_array(int total) {
    double log_total = log2(total);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, std::nextafter(log_total, std::numeric_limits<double>::max()));
    std::array<double, n> data;
    for (int i = 0; i < n - 1; i++) {
        data[i] = dis(gen);
    }
    data[n - 1] = log_total;
    std::sort(data.begin(), data.end());
    std::array<int, n> result;
    int tot = 1;
    for (int i = 0; i < n; i++) {
        result[i] = ceil(exp2(data[i]) / tot);
        tot *= result[i];
    }
    return result;
}

inline int random_int(int mx) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, mx);
    return dis(gen);
}

}


#endif //IR_UTILS_H