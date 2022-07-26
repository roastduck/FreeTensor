#ifndef FREE_TENSOR_RAND_COND_H
#define FREE_TENSOR_RAND_COND_H

#include <iostream>
#include <string>

#include <hash_combine.h>
#include <shared_linked_list.h>

namespace freetensor {

/**
 * Condition of a random variable
 */
struct RandCond {
    std::string name_;
    int value_;

    friend bool operator==(const RandCond &, const RandCond &) = default;

    friend std::ostream &operator<<(std::ostream &os, const RandCond &cond) {
        return os << cond.name_ << " = " << cond.value_;
    }
};

typedef SharedLinkedList<RandCond> RandCondStack;

/**
 * RAII guard to add a RandCond to a RandCondStack
 */
class RandCondGuard {
    RandCondStack &stack_;

  public:
    RandCondGuard(RandCondStack &stack, const std::string &name, int value)
        : stack_(stack) {
        stack_ = stack_.push(RandCond{name, value});
    }

    ~RandCondGuard() { stack_ = stack_.pop(); }
};

} // namespace freetensor

namespace std {

template <> class hash<freetensor::RandCond> {
    std::hash<std::string> hashName_;
    std::hash<int> hashValue_;

  public:
    size_t operator()(const freetensor::RandCond &cond) {
        return freetensor::hashCombine(hashName_(cond.name_),
                                       hashValue_(cond.value_));
    }
};

} // namespace std

#endif // FREE_TENSOR_RAND_COND_H
