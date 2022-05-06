#ifndef FREE_TENSOR_BUFFER_H
#define FREE_TENSOR_BUFFER_H

#include <array>
#include <string>

#include <itertools.hpp>

#include <container_utils.h>
#include <sub_tree.h>
#include <tensor.h>

namespace freetensor {

enum class AccessType : size_t {
    Input = 0,
    Output,
    InOut,
    Cache,
    // ------
    NumTypes,
};

// First deduce array length, then assert, to ensure the length
constexpr std::array accessTypeNames = {
    "input",
    "output",
    "inout",
    "cache",
};
static_assert(accessTypeNames.size() == (size_t)AccessType::NumTypes);

inline std::string toString(AccessType atype) {
    return accessTypeNames.at((size_t)atype);
}

inline AccessType parseAType(const std::string &_str) {
    auto &&str = tolower(_str);
    for (auto &&[i, s] : iter::enumerate(accessTypeNames)) {
        if (s == str) {
            return (AccessType)i;
        }
    }
    std::string msg = "Unrecognized access type \"" + _str +
                      "\". Candidates are (case-insensitive): ";
    for (auto &&[i, s] : iter::enumerate(accessTypeNames)) {
        msg += (i > 0 ? ", " : "");
        msg += s;
    }
    ERROR(msg);
}

enum class MemType : size_t {
    ByValue = 0, // Passed by value. Always in stack or registers
    CPU,         // Main memory
    GPUGlobal,
    GPUShared,
    GPULocal,
    GPUWarp,
    // ------
    NumTypes,
};

// First deduce array length, then assert, to ensure the length
constexpr std::array memTypeNames = {
    "byvalue", "cpu", "gpu/global", "gpu/shared", "gpu/local", "gpu/warp",
};
static_assert(memTypeNames.size() == (size_t)MemType::NumTypes);

inline std::string toString(MemType mtype) {
    return memTypeNames.at((size_t)mtype);
}

inline MemType parseMType(const std::string &_str) {
    auto &&str = tolower(_str);
    for (auto &&[i, s] : iter::enumerate(memTypeNames)) {
        if (s == str) {
            return (MemType)i;
        }
    }
    std::string msg = "Unrecognized memory type \"" + _str +
                      "\". Candidates are (case-insensitive): ";
    for (auto &&[i, s] : iter::enumerate(memTypeNames)) {
        msg += (i > 0 ? ", " : "");
        msg += s;
    }
    ERROR(msg);
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

} // namespace freetensor

#endif // FREE_TENSOR_BUFFER_H
