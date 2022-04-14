#ifndef IR_UTILS_H
#define IR_UTILS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <random>

namespace ir {

inline int randomInt(int mx, std::default_random_engine &gen) {
    std::uniform_int_distribution<> dis(0, mx);
    return dis(gen);
}

inline double randomDouble(std::default_random_engine &gen) {
    std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

inline std::vector<double> getProbSum(const std::vector<double> &pred) {
    std::vector<double> sum = pred;
    int sz = sum.size();
    sum[0] = (sum[0] > 1e20 ? 0 : 1 / sum[0]);
    for (int i = 1; i < sz; i++) {
        sum[i] = sum[i - 1] + (sum[i] > 1e20 ? 0 : 1 / sum[i]);
    }
    for (int i = 0; i < sz; i++) {
        sum[i] /= sum[sz - 1];
    }
    return sum;
}

inline int randWithProb(const std::vector<double> &probSum,
                        std::default_random_engine &gen) {
    std::uniform_real_distribution<> dis(0, 1);
    return std::upper_bound(probSum.begin(), probSum.end(), dis(gen)) -
           probSum.begin();
}

inline std::vector<int> randomFillArray(int total, int n,
                                        std::default_random_engine &gen) {
    int log_total = log2(total);
    std::vector<int> data(n, 1);
    for (int i = 0; i < log_total; i++) {
        data[randomInt(n - 1, gen)] *= 2;
    }
    return data;
}

} // namespace ir

#endif // IR_UTILS_H