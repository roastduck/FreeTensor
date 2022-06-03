#include <auto_schedule/factor_splitter.h>
#include <cmath>

namespace freetensor {

std::map<std::pair<int, int>, FactorSplitter::ResultType> FactorSplitter::results_;
std::map<int, std::vector<int>> FactorSplitter::factors_;
std::mutex FactorSplitter::mtxResults_;
std::mutex FactorSplitter::mtxFactors_;

const FactorSplitter::ResultType &FactorSplitter::get(int len, int n) {
    if (auto iter = results_.find({len, n}); iter != results_.end()) {
        return iter->second;
    }
    if (n == 1) {
        ResultType ret{{len}};
        mtxResults_.lock();
        if (!results_.count({len, n})) {
            results_[{len, n}] = std::move(ret);
        }
        mtxResults_.unlock();
        return results_[{len, n}];
    }
    ResultType ret;
    for (auto i : getFactors(len)) {
        const auto &r = get(len / i, n - 1);
        for (auto j : r) {
            j.push_back(i);
            ret.emplace_back(std::move(j));
        }
    }
    mtxResults_.lock();
    if (!results_.count({len, n})) {
        results_[{len, n}] = std::move(ret);
    }
    mtxResults_.unlock();
    return results_[{len, n}];
}

const std::vector<int> &FactorSplitter::getFactors(int n) {
    if (auto iter = factors_.find(n); iter != factors_.end()) {
        return iter->second;
    }
    std::vector<int> ret;
    for (int i = 1; i <= int(sqrt(n)); i++) {
        if (n % i == 0) {
            ret.push_back(i);
            if (n / i != i) {
                ret.push_back(n / i);
            }
        }
    }
    mtxFactors_.lock();
    if (!factors_.count(n)) {
        factors_[n] = std::move(ret);
    }
    mtxFactors_.unlock();
    return factors_[n];
}

} // namespace freetensor