#ifndef FREE_TENSOR_UTILS_H
#define FREE_TENSOR_UTILS_H

#include <algorithm>
#include <array>
#include <auto_schedule/factor_splitter.h>
#include <cmath>
#include <cstdlib>
#include <random>
#include <schedule.h>

namespace freetensor {

template <class RNG>
requires std::uniform_random_bit_generator<RNG> std::vector<int>
_randomFillArray(int total, int n, RNG &gen) {
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

template <class RNG>
requires std::uniform_random_bit_generator<RNG>
int randomInt(int mx, RNG &gen) {
    std::uniform_int_distribution<> dis(0, mx);
    return dis(gen);
}

template <class RNG>
requires std::uniform_random_bit_generator<RNG>
double randomDouble(RNG &gen) {
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

template <class RNG>
requires std::uniform_random_bit_generator<RNG>
int randWithProb(const std::vector<double> &probSum, RNG &gen) {
    std::uniform_real_distribution<> dis(0, 1);
    return std::upper_bound(probSum.begin(), probSum.end(), dis(gen)) -
           probSum.begin();
}

template <class RNG>
requires std::uniform_random_bit_generator<RNG> std::vector<int>
randomFillArray(int total, int n, RNG &gen) {
    const auto &candidates = FactorSplitter::get(total, n);
    return candidates[randomInt(candidates.size() - 1, gen)];
}

inline ID mergeLoops(Schedule &schedule, std::vector<ID> loops) {
    if (loops.empty()) {
        return {};
    }
    ID outermost = loops[0];
    for (size_t i = 1; i < loops.size(); i++) {
        outermost = schedule.merge(outermost, loops[i]);
    }
    return outermost;
}

inline std::vector<std::pair<ID, int>> splitLoop(Schedule &schedule, ID loop,
                                                 std::vector<int> tiling) {
    int n = tiling.size();
    std::vector<std::pair<ID, int>> result(n);
    for (int i = 0; i < n - 1; i++) {
        if (tiling[i] != 1) {
            auto t = schedule.split(loop, tiling[i]);
            loop = t.first;
            result[n - i - 1] = std::make_pair(t.second, tiling[i]);
        } else {
            result[n - i - 1] = std::make_pair("", 1);
        }
    }
    result[0] = std::make_pair(loop, tiling[n - 1]);
    return result;
}

inline Schedule::IDMap fissionLoops(Schedule &schedule,
                                    const std::vector<ID> &loops, ID splitter) {
    int n = loops.size();
    Schedule::IDMap ret;
    for (int i = n - 1; i >= 0; i--) {
        auto now =
            schedule.fission(loops[i], FissionSide::After, splitter).second;
        if (ret.empty()) {
            ret = now;
        } else {
            for (auto &&[key, value] : ret) {
                ret[key] = now[value];
            }
            ret[loops[i]] = loops[i].strId() + ".b";
        }
        splitter = loops[i].strId() + ".a";
    }
    return ret;
}

} // namespace freetensor

#endif // FREE_TENSOR_UTILS_H
