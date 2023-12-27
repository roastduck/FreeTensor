#ifndef FREE_TENSOR_BUFFER_H
#define FREE_TENSOR_BUFFER_H

#include <sub_tree.h>
#include <tensor.h>
#include <type/access_type.h>
#include <type/mem_type.h>

namespace freetensor {

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

} // namespace freetensor

#endif // FREE_TENSOR_BUFFER_H
