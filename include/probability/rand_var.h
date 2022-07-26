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
 * using Bayesian statistics
 *
 * The more observations is observed for a value, the value will have a higher
 * probability of being sampled. Specifically, when sampled, the distribution
 * follows a Categorical (or Bernoulli for n=2) distribution, where the
 * probability of the i-th value is `P(X=i) = p_i`. `p_i` itself is another
 * random variables follows Dirichlet (or Beta for n=2) distribution `p_i ~
 * Dir(alpha)`, where `alpha` is the observations + 1
 */
class DiscreteRandVar {
    std::string name_;
    RandCondStack conds_;
    std::vector<int> obs_;

  private:
    template <std::uniform_random_bit_generator RNG>
    std::vector<double> dirichletSample(RNG &rng) const {
        // Compute Dirichlet distribution via Gamma distribution
        // (https://en.wikipedia.org/wiki/Dirichlet_distribution#Computational_methods)
        auto n = obs_.size();
        std::vector<double> ret(n);
        for (size_t i = 0; i < n; i++) {
            ret[i] = std::gamma_distribution<double>{obs_[i] + 1., 1.}(rng);
        }
        // std::discrete_distribution will normalize it for us
        return ret;
    }

  public:
    DiscreteRandVar(const std::string &name, const RandCondStack &conds,
                    const std::vector<int> &initObs)
        : name_(name), conds_(conds), obs_(initObs) {}

    void observe(int value, int cnt = 1) { obs_.at(value) += cnt; }

    template <std::uniform_random_bit_generator RNG>
    int sample(RNG &rng) const {
        auto discreteProb = dirichletSample(rng);
        return std::discrete_distribution<int>(discreteProb.begin(),
                                               discreteProb.end())(rng);
    }

    int mostLikely() const {
        return std::max_element(obs_.begin(), obs_.end()) - obs_.begin();
    }

    /**
     * Clone as an indenpenden variable
     */
    Ref<DiscreteRandVar> clone() const {
        return Ref<DiscreteRandVar>::make(*this);
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
    Ref<DiscreteRandVar> var_;
    Ref<DiscreteRandVar> varSnapshot_; // Freeze the distribution at the time of
                                       // observation, used for debugging
    int value_;
    std::string message_; // Debug info

    DiscreteObservation(const Ref<DiscreteRandVar> &var, int value,
                        const std::string &message = "")
        : var_(var), varSnapshot_(var->clone()), value_(value),
          message_(message) {}

    friend auto operator<=>(const DiscreteObservation &lhs,
                            const DiscreteObservation &rhs) {
        if (auto cmp = lhs.var_ <=> rhs.var_; cmp != 0) {
            return cmp;
        }
        return lhs.value_ <=> rhs.value_;
    }
    friend bool operator==(const DiscreteObservation &lhs,
                           const DiscreteObservation &rhs) {
        return lhs.var_ == rhs.var_ && lhs.value_ == rhs.value_;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const DiscreteObservation &obs);
};

} // namespace freetensor

#endif // FREE_TENSOR_RAND_VAR_H
