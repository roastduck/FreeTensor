#ifndef FREE_TENSOR_UTILS_H
#define FREE_TENSOR_UTILS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <random>

namespace freetensor {

inline std::vector<int> _randomFillArray(int total, int n,
                                         std::default_random_engine &gen) {
    double log_total = log2(total);
    std::uniform_real_distribution<> dis(
        0, std::nextafter(log_total, std::numeric_limits<double>::max()));
    std::vector<double> data;
    data.reserve(n);
    for (int i = 0; i < n - 1; i++) {
        data.push_back(dis(gen));
    }
    data.push_back(log_total);
    std::sort(data.begin(), data.end());
    std::vector<int> result;
    result.reserve(n);
    int tot = 1;
    for (int i = 0; i < n; i++) {
        result.push_back(ceil(exp2(data[i]) / tot));
        tot *= result[i];
    }
    return result;
}

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
    for (int i = 0; i < sz; i++) {
        if (sum[i] > -1e20) {
            sum[i] = std::max(sum[i], 1.);
        }
    }
    sum[0] = (sum[0] <= -1e20 ? 0 : sum[0]);
    for (int i = 1; i < sz; i++) {
        sum[i] = sum[i - 1] + (sum[i] <= -1e20 ? 0 : sum[i]);
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

} // namespace freetensor

#endif // FREE_TENSOR_UTILS_H
