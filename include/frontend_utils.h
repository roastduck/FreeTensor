#ifndef FRONTEND_UTILIS
#define FRONTEND_UTILIS

#include <debug.h>
#include <expr.h>
#include <stmt.h>

namespace ir {

enum class FrontendVarIdxType : int { Single, Slice };

class FrontendVarIdx {
    FrontendVarIdxType type_;
    Expr start_, stop_;

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

    static FrontendVarIdx fromSingle(const Expr &single) {
        FrontendVarIdx ret;
        ret.type_ = FrontendVarIdxType::Single;
        ret.start_ = single;
        return ret;
    }

    static FrontendVarIdx fromSlice(const Expr &start, const Expr &stop) {
        FrontendVarIdx ret;
        ret.type_ = FrontendVarIdxType::Slice;
        ret.start_ = start;
        ret.stop_ = stop;
        return ret;
    }
};

inline std::string toString(const FrontendVarIdx &idx) {
    if (idx.type() == FrontendVarIdxType::Single) {
        return toString(idx.single());
    } else {
        return "(" + toString(idx.start()) + ", " + toString(idx.stop()) + ")";
    }
}

class FrontendVar {
    std::string name_;
    std::vector<Expr> fullShape_;
    DataType dtype_;
    MemType mtype_;
    std::vector<FrontendVarIdx> indices_;

  public:
    FrontendVar(const std::string &name, const std::vector<Expr> &fullShape,
                DataType dtype, MemType mtype,
                const std::vector<FrontendVarIdx> &indices)
        : name_(name), fullShape_(fullShape), dtype_(dtype), mtype_(mtype),
          indices_(indices) {}

    const std::string &name() const { return name_; }

    /**
     * shape of the original variable before slicing
     */
    const std::vector<Expr> &fullShape() const { return fullShape_; }

    DataType dtype() const { return dtype_; }
    MemType mtype() const { return mtype_; }
    int ndim() const;
    const std::vector<FrontendVarIdx> &indices() const { return indices_; }

    /**
     * shape of each dimension after slicing
     */
    Expr shape(const Expr &idx) const;

    Expr asLoad() const;
    Stmt asStore(const std::string &id, const Expr &value) const;

    std::vector<FrontendVarIdx>
    chainIndices(const std::vector<FrontendVarIdx> &next) const;
};

inline std::string toString(const FrontendVar &var) {
    std::string ret = var.name() + "[";
    bool first = true;
    for (auto &&idx : var.indices()) {
        if (!first) {
            ret += ", ";
        }
        ret += toString(idx);
    }
    ret += "]";
    return ret;
}

} // namespace ir

#endif // FRONTEND_UTILS
