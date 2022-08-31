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

inline std::vector<int>
_randomFillArray(int total, int n,
                 std::uniform_random_bit_generator auto &gen) {
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

inline int randomInt(int mx, std::uniform_random_bit_generator auto &gen) {
    std::uniform_int_distribution<> dis(0, mx);
    return dis(gen);
}

inline std::vector<double> getProbFromPredict(const std::vector<double> &pred) {
    std::vector<double> prob;
    prob.reserve(pred.size());
    for (auto x : pred) {
        if (x > -1e20) {
            // Valid sketch gets at least weight 1 to pick
            prob.emplace_back(std::max(x, 1.));
        } else {
            // Invalid sketch gets no chance to pick
            prob.emplace_back(0);
        }
    }
    return prob;
}

inline std::vector<int>
randomFillArray(int total, int n, std::uniform_random_bit_generator auto &gen) {
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
            result[n - i - 1] = {t.second, tiling[i]};
        } else {
            result[n - i - 1] = {{}, 1};
        }
    }
    result[0] = {loop, tiling[n - 1]};
    return result;
}

inline Schedule::IDMap fissionLoops(Schedule &schedule,
                                    const std::vector<ID> &loops, ID splitter) {
    int n = loops.size();
    Schedule::IDMap ret;
    for (int i = n - 1; i >= 0; i--) {
        auto [before, after] =
            schedule.fission(loops[i], FissionSide::After, splitter);
        if (ret.empty()) {
            ret = after;
        } else {
            for (auto &&[key, value] : ret) {
                ret[key] = after[value];
            }
            ret[loops[i]] = after[loops[i]];
        }
        ASSERT(before.count(loops[i]) == 1);
        splitter = before[loops[i]];
    }
    return ret;
}

} // namespace freetensor

#endif // FREE_TENSOR_UTILS_H
