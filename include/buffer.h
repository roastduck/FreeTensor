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

enum class MemType : int { CPU, GPUGlobal, GPUShared, GPULocal };

inline std::string toString(MemType mtype) {
    switch (mtype) {
    case MemType::CPU:
        return "[CPU]";
    case MemType::GPUGlobal:
        return "[GPUGlobal]";
    case MemType::GPUShared:
        return "[GPUShared]";
    case MemType::GPULocal:
        return "[GPULocal]";
    }
    return "[???]";
}

class Buffer {
    Tensor tensor_;
    AccessType atype_;
    MemType mtype_;

  public:
    Buffer(Tensor &&tensor, AccessType atype, MemType mtype)
        : tensor_(std::move(tensor)), atype_(atype), mtype_(mtype) {}
    Buffer(const Tensor &tensor, AccessType atype, MemType mtype)
        : tensor_(tensor), atype_(atype), mtype_(mtype) {}

    const Tensor &tensor() const { return tensor_; }
    Tensor &tensor() { return tensor_; }

    void setAtype(AccessType atype) { atype_ = atype; }
    AccessType atype() const { return atype_; }

    void setMtype(MemType mtype) { mtype_ = mtype; }
    MemType mtype() const { return mtype_; }
};

} // namespace ir

#endif // BUFFER_H
