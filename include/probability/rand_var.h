#ifndef FREE_TENSOR_RAND_VAR_H
#define FREE_TENSOR_RAND_VAR_H

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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
    std::vector<int> obs_;

  private:
    template <std::uniform_random_bit_generator RNG>
    std::vector<double> dirichletSample(RNG &rng) const {
        // Compute Dirichlet distribution via Gamma distribution
        // (https://en.wikipedia.org/wiki/Dirichlet_distribution#Computational_methods)
        auto n = obs_.size();
        std::vector<double> ret(n);
        double sum = 0;
        for (size_t i = 0; i < n; i++) {
            ret[i] = std::gamma_distribution<double>{obs_[i] + 1., 1.}(rng);
            sum += ret[i];
        }
        for (size_t i = 0; i < n; i++) {
            ret[i] /= sum;
        }
        return ret;
    }

  public:
    DiscreteRandVar(const std::string &name, const std::vector<int> &initObs)
        : name_(name), obs_(initObs) {}

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

    DiscreteObservation(const Ref<DiscreteRandVar> &var, int value)
        : var_(var), varSnapshot_(var->clone()), value_(value) {}

    friend auto operator<=>(const DiscreteObservation &,
                            const DiscreteObservation &) = default;
    friend bool operator==(const DiscreteObservation &,
                           const DiscreteObservation &) = default;

    friend std::ostream &operator<<(std::ostream &os,
                                    const DiscreteObservation &obs) {
        return os << "(" << *obs.varSnapshot_ << ") = " << obs.value_;
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_RAND_VAR_H
