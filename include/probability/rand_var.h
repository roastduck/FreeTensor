#ifndef FREE_TENSOR_RAND_VAR_H
#define FREE_TENSOR_RAND_VAR_H

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <probability/rand_cond.h>
#include <ref.h>

namespace freetensor {

/**
 * Discrete random variable, can be any integer in `[0, n)`
 *
 * A random variable that can be sampled (infered) and observed (learnt)
 * using Naive Bayes
 *
 * In Naive Bayes, the probability of an event `C` given conditions `F_i`, can
 * be calculated by `p(C) \Prod_i P(F_i | C)`
 * (https://en.wikipedia.org/wiki/Naive_Bayes_classifier). Here a
 * `DiscreteRandVar` records `P(F_i | C)` for each `C` as `prob`. When picking
 * the most likely `C`, one only need to multiply the `obsVec` vectors from each
 * `F_i`, and pick a `C` with highest value
 */
class DiscreteRandVar {
    std::string name_;
    Ref<RandCondInterface> cond_;
    Ref<std::vector<int>> totCnt_;
    std::vector<int> obs_;

  public:
    DiscreteRandVar(const std::string &name, const Ref<RandCondInterface> &cond,
                    const Ref<std::vector<int>> totCnt,
                    const std::vector<int> &initObs)
        : name_(name), cond_(cond), totCnt_(totCnt), obs_(initObs) {}

    void observe(int value, int cnt = 1) { obs_.at(value) += cnt; }

    std::vector<double> prob() const {
        std::vector<double> ret;
        ret.reserve(obs_.size());
        for (auto &&[p, q] : views::zip(obs_, *totCnt_)) {
            ret.emplace_back((double)p / q);
        }
        return ret;
    }

    /**
     * Clone as an indenpenden variable
     */
    Ref<DiscreteRandVar> clone() const {
        auto ret = Ref<DiscreteRandVar>::make(*this);
        ret->totCnt_ = Ref<std::vector<int>>::make(*ret->totCnt_);
        return ret;
    }

    const std::string &name() const { return name_; }

    friend std::ostream &operator<<(std::ostream &os,
                                    const DiscreteRandVar &var);
};

/**
 * An observation to a discreteRandVar
 *
 * The `DiscreteRandVar` is identified by its address: The same address means
 * the same `DiscreteRandVar`; Different addresses means independent
 * `DiscreteRandVar`s
 */
struct DiscreteObservation {
    std::vector<Ref<DiscreteRandVar>> vars_;
    std::vector<Ref<DiscreteRandVar>>
        varsSnapshot_; // Freeze the distribution at the time of
                       // observation, used for debugging
    Ref<std::vector<int>> totCnt_;
    int value_;
    std::string message_; // Debug info

    DiscreteObservation(const std::vector<Ref<DiscreteRandVar>> &vars,
                        const Ref<std::vector<int>> &totCnt, int value,
                        const std::string &message = "")
        : vars_(vars), totCnt_(totCnt), value_(value), message_(message) {
        varsSnapshot_.reserve(vars_.size());
        for (auto &&var : vars_) {
            varsSnapshot_.emplace_back(var->clone());
        }
    }

    friend auto operator<=>(const DiscreteObservation &lhs,
                            const DiscreteObservation &rhs) {
        if (auto cmp = lhs.vars_ <=> rhs.vars_; cmp != 0) {
            return cmp;
        }
        return lhs.value_ <=> rhs.value_;
    }
    friend bool operator==(const DiscreteObservation &lhs,
                           const DiscreteObservation &rhs) {
        return lhs.vars_ == rhs.vars_ && lhs.value_ == rhs.value_;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const DiscreteObservation &obs);
};

} // namespace freetensor

#endif // FREE_TENSOR_RAND_VAR_H
