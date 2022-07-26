#include <container_utils.h>
#include <probability/rand_var.h>

namespace freetensor {

std::ostream &operator<<(std::ostream &os, const DiscreteRandVar &var) {
    os << "P(" << var.name_;
    if (!var.conds_.empty()) {
        os << " | " << var.conds_.asVector();
    }
    os << ") ~ Bernoulli(p ~ Dir({" << var.obs_ << "} + 1))";
    return os;
}

std::ostream &operator<<(std::ostream &os, const DiscreteObservation &obs) {
    os << "(" << *obs.varSnapshot_ << ") = " << obs.value_;
    if (!obs.message_.empty()) {
        os << " /* " << obs.message_ << " */";
    }
    return os;
}

} // namespace freetensor
