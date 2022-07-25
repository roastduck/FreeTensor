#ifndef FREE_TENSOR_RAND_CTX_H
#define FREE_TENSOR_RAND_CTX_H

#include <map>
#include <regex>
#include <string>
#include <unordered_map>

#include <func_utils.h>
#include <probability/rand_var.h>
#include <shared_linked_list.h>

namespace freetensor {

struct ProgramPositionHelper {};
typedef const ProgramPositionHelper *ProgramPosition;

/**
 * Uniquely marking a program position
 */
#define PROGRAM_POSITION                                                       \
    ([]() -> ProgramPosition {                                                 \
        static ProgramPositionHelper p;                                        \
        return &p;                                                             \
    })()

typedef std::vector<DiscreteObservation> RandTrace;

/**
 * Non-template base class for `RandCtx`
 */
class RandCtxImpl {
  protected:
    SharedLinkedList<int> condStack_;
    std::unordered_map<
        ProgramPosition,
        std::unordered_map<SharedLinkedList<int>, Ref<DiscreteRandVar>>>
        randVars_;

    std::multimap<Ref<RandTrace>, std::pair<double, double>,
                  PtrInvocable<std::less<RandTrace>>>
        traces_; // Ordered map

    bool isInfer_ = true;
    std::regex toLearn_{".*"};

  public:
    void pushCond(int value) { condStack_ = condStack_.push(value); }

    void popCond() { condStack_ = condStack_.pop(); }

    void observeTrace(const Ref<RandTrace> &trace, double value, double stddev);

    /**
     * Set to learn some of the random variables only
     *
     * A pattern can be set, and then only the variables matching this pattern
     * is set to learning. This is useful when testing some of the random
     * decisions with few learning trials
     */
    void setLearnFilter(const std::regex &toLearn) { toLearn_ = toLearn; }

    /**
     * When `decide` later, randomly sample a decision
     */
    void setLearn() { isInfer_ = false; }

    /**
     * When `decide` later, pick a most likely decision
     */
    void setInfer() { isInfer_ = true; }
};

/**
 * Context to do random decisions
 *
 * If `setLearn`ed, sample a decision and use the decision traces to learn a
 * Bayesian model. Each trace is labeled with a performance value, and traces
 * are compared pairwisely (lower is better), to learn P(this decision leads to
 * better performance).
 */
template <std::uniform_random_bit_generator RNG>
class RandCtx : public RandCtxImpl {
    RNG &rng_;

  public:
    RandCtx(RNG &rng) : rng_(rng) {}

    /**
     * Get decision from a random variable uniquely defined by conditions and
     * the program position
     *
     * If learning, sample a random variable and record it as a trace. If
     * infering, pick a most likely decision
     */
    int decide(ProgramPosition pos, const std::string &name,
               const std::vector<int> &initObs, const Ref<RandTrace> &trace) {
        if (!randVars_.count(pos) || !randVars_.at(pos).count(condStack_)) {
            randVars_[pos][condStack_] =
                Ref<DiscreteRandVar>::make(name, initObs);
        }
        auto &&var = randVars_.at(pos).at(condStack_);
        auto value = isInfer_ || !std::regex_match(var->name(), toLearn_)
                         ? var->mostLikely()
                         : var->sample(rng_);
        if (trace.isValid()) {
            trace->emplace_back(var, value);
        }
        return value;
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_RAND_CTX_H
