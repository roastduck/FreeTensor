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
    template <class T> friend Ref<Buffer> makeBuffer(T &&, AccessType, MemType);

    SubTree<Tensor> tensor_ = ChildOf{this};
    AccessType atype_;
    MemType mtype_;

  public:
    const auto &tensor() const { return tensor_; }
    auto &tensor() { return tensor_; }

    void setAtype(AccessType atype) { atype_ = atype; }
    AccessType atype() const { return atype_; }

    void setMtype(MemType mtype) { mtype_ = mtype; }
    MemType mtype() const { return mtype_; }

    void compHash() override;
};

template <class T>
Ref<Buffer> makeBuffer(T &&tensor, AccessType atype, MemType mtype) {
    auto b = Ref<Buffer>::make();
    b->tensor_ = std::forward<T>(tensor);
    b->atype_ = atype;
    b->mtype_ = mtype;
    return b;
}

inline Ref<Buffer> deepCopy(const Ref<Buffer> &b) {
    return makeBuffer(b->tensor(), b->atype(), b->mtype());
}

} // namespace ir

#endif // BUFFER_H
