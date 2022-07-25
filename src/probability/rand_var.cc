#include <container_utils.h>
#include <probability/rand_var.h>

namespace freetensor {

std::ostream &operator<<(std::ostream &os, const DiscreteRandVar &var) {
    // TODO: Pring conditions
    return os << "P(" << var.name_ << ") ~ Bernoulli(p ~ Dir({" << var.obs_
              << "} + 1))";
}

} // namespace freetensor
