#ifndef FREE_TENSOR_FRONTEND_VAR
#define FREE_TENSOR_FRONTEND_VAR

#include <iostream>

#include <container_utils.h>
#include <debug.h>
#include <expr.h>
#include <pass/const_fold.h>
#include <serialize/to_string.h>
#include <stmt.h>

namespace freetensor {

enum class FrontendVarIdxType : int { Single, Slice };

class FrontendVarIdx {
    FrontendVarIdxType type_;
    Expr start_, stop_, len_;

  public:
    FrontendVarIdxType type() const { return type_; }

    const Expr &single() const {
        ASSERT(type_ == FrontendVarIdxType::Single);
        return start_;
    }

    const Expr &start() const {
        ASSERT(type_ == FrontendVarIdxType::Slice);
        return start_;
    }

    const Expr &stop() const {
        ASSERT(type_ == FrontendVarIdxType::Slice);
        return stop_;
    }

    const Expr &len() const { return len_; }

    static FrontendVarIdx fromSingle(const Expr &single) {
        FrontendVarIdx ret;
        ret.type_ = FrontendVarIdxType::Single;
        ret.start_ = constFold(single);
        ret.len_ = makeIntConst(1);
        return ret;
    }

    static FrontendVarIdx fromSlice(const Expr &start, const Expr &stop,
                                    const Expr &len = nullptr) {
        FrontendVarIdx ret;
        ret.type_ = FrontendVarIdxType::Slice;
        if (start.isValid()) {
            ret.start_ = start;
        } else {
            ASSERT(stop.isValid());
            ASSERT(len.isValid());
            ret.start_ = constFold(makeSub(stop, len));
        }
        if (stop.isValid()) {
            ret.stop_ = stop;
        } else {
            ASSERT(start.isValid());
            ASSERT(len.isValid());
            ret.stop_ = constFold(makeAdd(start, len));
        }
        if (len.isValid()) {
            ret.len_ = len;
        } else {
            ASSERT(start.isValid());
            ASSERT(stop.isValid());
            ret.len_ = constFold(makeSub(stop, start));
        }
        return ret;
    }
};

inline std::ostream &operator<<(std::ostream &os, const FrontendVarIdx &idx) {
    if (idx.type() == FrontendVarIdxType::Single) {
        return os << idx.single();
    } else {
        return os << "(" << idx.start() << ", " << idx.stop() << ")";
    }
}

class FrontendVar {
    std::string name_;
    std::vector<Expr> fullShape_;
    DataType dtype_;
    MemType mtype_;
    std::vector<FrontendVarIdx> indices_;
    bool isLoadAtVersion_;

  public:
    FrontendVar(const std::string &name, const std::vector<Expr> &fullShape,
                DataType dtype, MemType mtype,
                const std::vector<FrontendVarIdx> &indices,
                bool isLoadAtVersion = false)
        : name_(name), fullShape_(fullShape), dtype_(dtype), mtype_(mtype),
          indices_(indices), isLoadAtVersion_(isLoadAtVersion) {}

    const std::string &name() const { return name_; }

    /**
     * shape of the original variable before slicing
     */
    const std::vector<Expr> &fullShape() const { return fullShape_; }

    DataType dtype() const { return dtype_; }
    MemType mtype() const { return mtype_; }

    /**
     * Number of dimensions after slicing
     */
    int ndim() const;

    const std::vector<FrontendVarIdx> &indices() const { return indices_; }

    /**
     * shape of each dimension after slicing
     * @{
     */
    Expr shape(const Expr &idx) const;
    std::vector<Expr> shape() const;
    /** @} */

    Expr asLoad() const;
    Stmt asStore(const Metadata &metadata, const Expr &value) const;
    Stmt asReduceTo(ReduceOp op, const Metadata &metadata, const Expr &value,
                    bool atomic = false) const;

    std::vector<FrontendVarIdx>
    chainIndices(const std::vector<FrontendVarIdx> &next) const;
};

inline std::ostream &operator<<(std::ostream &os, const FrontendVar &var) {
    os << var.name() << "[";
    for (auto &&[i, idx] : views::enumerate(var.indices())) {
        os << (i == 0 ? "" : ", ") << idx;
    }
    return os << "]";
}

std::unordered_set<std::string> allReads(const FrontendVarIdx &idx);

} // namespace freetensor

#endif // FREE_TENSOR_FRONTEND_VAR
