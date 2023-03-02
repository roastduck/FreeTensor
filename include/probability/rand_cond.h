#ifndef FREE_TENSOR_RAND_COND_H
#define FREE_TENSOR_RAND_COND_H

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>

#include <debug.h>
#include <hash_combine.h>
#include <shared_linked_list.h>

namespace freetensor {

/**
 * Condition of a random variable
 */
class RandCondInterface {
  public:
    virtual ~RandCondInterface() {}
    virtual std::type_index typeId() const = 0;
    virtual std::string name() const = 0;
    virtual size_t hash() const = 0;
    virtual bool sameAs(const RandCondInterface &other) const = 0;
    virtual std::string toString() const = 0;

    friend bool operator==(const RandCondInterface &lhs,
                           const RandCondInterface &rhs) {
        return lhs.sameAs(rhs);
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const RandCondInterface &self) {
        return os << self.toString();
    }
};

template <class Derived> class RandCondCRTP : public RandCondInterface {
    std::type_index typeId_;

  public:
    RandCondCRTP() : typeId_(typeid(Derived)) {}

    std::type_index typeId() const override final { return typeId_; }

    bool sameAs(const RandCondInterface &other) const override final {
        if (typeId_ != other.typeId()) {
            return false;
        }
        return (const Derived &)(*this) == (const Derived &)other;
    };
};

template <class T, class Hasher = std::hash<T>,
          class Comparator = std::equal_to<T>>
class RandCond : public RandCondCRTP<RandCond<T, Hasher, Comparator>> {
    std::string name_;
    T value_;

  public:
    RandCond(const std::string &name, const T &value)
        : name_(name), value_(value) {}

    std::string name() const override { return name_; }

    size_t hash() const override {
        return hashCombine(std::hash<std::string>{}(name_), Hasher{}(value_));
    }

    friend bool operator==(const RandCond &lhs, const RandCond &rhs) {
        return lhs.name_ == rhs.name_ && Comparator{}(lhs.value_, rhs.value_);
    }

    std::string toString() const override {
        std::ostringstream os;
        os << name_ << " = " << value_;
        return os.str();
    }
};

typedef SharedLinkedList<Ref<RandCondInterface>> RandCondStack;

/**
 * RAII guard to add a RandCond to a RandCondStack
 */
template <class T, class Hasher = std::hash<T>,
          class Comparator = std::equal_to<T>>
class RandCondGuard {
    RandCondStack &stack_;

  public:
    RandCondGuard(RandCondStack &stack, const std::string &name, const T &value)
        : stack_(stack) {
        stack_ = stack_.push(
            Ref<RandCond<T, Hasher, Comparator>>::make(name, value));
    }

    ~RandCondGuard() { stack_ = stack_.pop(); }
};

} // namespace freetensor

namespace std {

template <> struct hash<freetensor::RandCondInterface> {
    size_t operator()(const freetensor::RandCondInterface &obj) const {
        return obj.hash();
    }
};

} // namespace std

#endif // FREE_TENSOR_RAND_COND_H
