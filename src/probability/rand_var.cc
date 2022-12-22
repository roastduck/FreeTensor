#include <container_utils.h>
#include <probability/rand_var.h>

namespace freetensor {

std::ostream &operator<<(std::ostream &os, const DiscreteRandVar &var) {
    os << "P(" << *var.cond_ << " | " << var.name_ << ") ~ Bernoulli({";
    for (auto &&[i, p, q] : views::zip(views::ints(0, ranges::unreachable),
                                       var.obs_, *var.totCnt_)) {
        os << (i > 0 ? ", " : "") << p << "/" << q;
    }
    return os << "})";
}

std::ostream &operator<<(std::ostream &os, const DiscreteObservation &obs) {
    os << "Considering the following:" << std::endl;
    for (auto &&var : obs.varsSnapshot_) {
        os << "  " << *var << std::endl;
    }
    os << "Decision ";
    if (!obs.message_.empty()) {
        os << "of \"" << obs.message_ << "\" ";
    }
    os << "is " << obs.value_;
    return os;
}

} // namespace freetensor
