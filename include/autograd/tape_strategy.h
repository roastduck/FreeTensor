#ifndef FREE_TENSOR_TAPE_STRATEGY_H
#define FREE_TENSOR_TAPE_STRATEGY_H

#include <unordered_set>
#include <variant>

#include <selector.h>
#include <stmt.h>

namespace freetensor {

/**
 * Mode of which intermediate variables should be stored.
 */
enum class GradTapeMode : int {
    All,        /// Store all variables including local scalars.
    Nothing,    /// Store nothing.
    NoReuseOnly /// Store variables that only hold one version of data, which
                /// means we do not have to store each version of them in their
                /// history.
};

class TapeStrategy {
    std::unordered_set<std::variant<ID, std::string, Ref<Selector>>>
        alwaysTape_, neverTape_;
    GradTapeMode mode_ = GradTapeMode::Nothing;

  public:
    TapeStrategy(
        const std::unordered_set<std::variant<ID, std::string, Ref<Selector>>>
            &tape)
        : alwaysTape_(tape) {}
    TapeStrategy(
        std::unordered_set<std::variant<ID, std::string, Ref<Selector>>> &&tape)
        : alwaysTape_(std::move(tape)) {}

    TapeStrategy(const std::ranges::range auto &r)
        : alwaysTape_(std::ranges::begin(r), std::ranges::end(r)) {}

    TapeStrategy(GradTapeMode mode) : mode_(mode) {}

    TapeStrategy alwaysTape(const std::ranges::range auto &r) {
        auto ret = *this;
        ret.alwaysTape_.insert(std::ranges::begin(r), std::ranges::end(r));
        return ret;
    }

    TapeStrategy neverTape(const std::ranges::range auto &r) {
        auto ret = *this;
        ret.neverTape_.insert(std::ranges::begin(r), std::ranges::end(r));
        return ret;
    }

    std::unordered_set<ID> getIdsToTape(const Stmt &ast) const;
};

} // namespace freetensor

#endif // FREE_TENSOR_TAPE_STRATEGY_H
