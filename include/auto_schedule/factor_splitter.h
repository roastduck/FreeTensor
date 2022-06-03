#ifndef FREE_TENSOR_FACTOR_SPLITTER_H
#define FREE_TENSOR_FACTOR_SPLITTER_H

#include <map>
#include <mutex>
#include <vector>

namespace freetensor {

class FactorSplitter {
    typedef std::vector<std::vector<int>> ResultType;
    static std::map<std::pair<int, int>, ResultType> results_;
    static std::map<int, std::vector<int>> factors_;
    static const std::vector<int> &getFactors(int n);
    static std::mutex mtxResults_;
    static std::mutex mtxFactors_;

  public:
    static const ResultType &get(int len, int n);
};

} // namespace freetensor

#endif // FREE_TENSOR_FACTOR_SPLITTER_H
