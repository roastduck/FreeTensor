#ifndef BUFFER_H
#define BUFFER_H

#include <string>

#include <tensor.h>

namespace ir {

enum class AccessType : int { Input, Output, InOut, Cache };

inline std::string toString(AccessType atype) {
    switch (atype) {
    case AccessType::Input:
        return "[in]";
    case AccessType::Output:
        return "[out]";
    case AccessType::InOut:
        return "[inout]";
    case AccessType::Cache:
        return "[cache]";
    }
    return "[???]";
}

class Buffer {
    Tensor tensor_;
    AccessType atype_;

  public:
    Buffer(Tensor &&tensor, AccessType atype)
        : tensor_(std::move(tensor)), atype_(atype) {}
    Buffer(const Tensor &tensor, AccessType atype)
        : tensor_(tensor), atype_(atype) {}

    const Tensor &tensor() const { return tensor_; }
    Tensor &tensor() { return tensor_; }

    void setAtype(AccessType atype) { atype_ = atype; }
    AccessType atype() const { return atype_; }
};

} // namespace ir

#endif // BUFFER_H
