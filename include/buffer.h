#ifndef BUFFER_H
#define BUFFER_H

#include <string>

#include <sub_tree.h>
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

enum class MemType : int {
    ByValue, // Passed by value. Always in stack or registers
    CPU,     // Main memory
    GPUGlobal,
    GPUShared,
    GPULocal,
    GPUWarp
};

inline std::string toString(MemType mtype) {
    switch (mtype) {
    case MemType::ByValue:
        return "[ByValue]";
    case MemType::CPU:
        return "[CPU]";
    case MemType::GPUGlobal:
        return "[GPUGlobal]";
    case MemType::GPUShared:
        return "[GPUShared]";
    case MemType::GPULocal:
        return "[GPULocal]";
    case MemType::GPUWarp:
        return "[GPUWarp]";
    }
    return "[???]";
}

class Buffer : public ASTPart {
    SubTree<Tensor> tensor_;
    AccessType atype_;
    MemType mtype_;

  public:
    template <class T>
    Buffer(T &&tensor, AccessType atype, MemType mtype)
        : tensor_(std::forward<T>(tensor)), atype_(atype), mtype_(mtype) {}

    const auto &tensor() const { return tensor_; }
    auto &tensor() { return tensor_; }

    void setAtype(AccessType atype) { atype_ = atype; }
    AccessType atype() const { return atype_; }

    void setMtype(MemType mtype) { mtype_ = mtype; }
    MemType mtype() const { return mtype_; }
};

inline Ref<Buffer> deepCopy(const Ref<Buffer> &b) {
    return Ref<Buffer>::make(*b);
}

} // namespace ir

#endif // BUFFER_H
