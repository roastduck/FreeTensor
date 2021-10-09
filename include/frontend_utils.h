#ifndef FRONTEND_UTILIS
#define FRONTEND_UTILIS

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

class FrontendVar {
    std::string name_;
    std::vector<Expr> shape_;
    DataType dtype_;
    std::vector<FrontendVarIdx> indices_;

  public:
    FrontendVar(const std::string &name, const std::vector<Expr> &shape,
                DataType dtype, const std::vector<FrontendVarIdx> &indices)
        : name_(name), shape_(shape), dtype_(dtype), indices_(indices) {}

    const std::string &name() const { return name_; }
    const std::vector<Expr> &shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    int ndim() const;
    const std::vector<FrontendVarIdx> &indices() const { return indices_; }

    Expr shapeAt(const Expr &idx) const;

    Expr asLoad() const;
    Stmt asStore(const std::string &id, const Expr &value) const;

    std::vector<FrontendVarIdx>
    chainIndices(const std::vector<FrontendVarIdx> &next) const;
};

} // namespace ir

#endif // FRONTEND_UTILS
