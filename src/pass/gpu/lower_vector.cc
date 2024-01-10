#ifdef FT_WITH_CUDA

#include <algorithm>

#include <hash.h>
#include <pass/gpu/lower_vector.h>
#include <pass/simplify.h>

namespace freetensor {

namespace gpu {

namespace {

class InvalidGPUVector : public InvalidProgram {
  public:
    InvalidGPUVector(const std::string &msg) : InvalidProgram(msg) {}
};

} // namespace

std::string LowerVector::vecType(DataType dtype) const {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types
    // TODO: Vectors of bool should be bitvectors
    std::string ret;
    switch (dtype.base()) {
    case DataType::Float64:
        ret = "double";
        break;
    case DataType::Float32:
        ret = "float";
        break;
    case DataType::Int64:
        ret = "longlong";
        break;
    case DataType::Int32:
        ret = "int";
        break;
    default:
        ERROR(FT_MSG << "Unsupported data type " << dtype);
    }
    return ret + std::to_string(vecLen_);
}

bool LowerVector::hasVectorIndices(const std::vector<Expr> &indices,
                                   const std::vector<Expr> &shape) {
    Expr index;
    for (auto &&[idx, dim] : views::zip(indices, shape)) {
        index = index.isValid() ? makeAdd(makeMul(index, dim), idx) : idx;
    }
    if (!index.isValid()) {
        return false;
    }

    // Any expression is analyzed as a linear expression
    analyzeLinear_(index);
    auto &&lin = analyzeLinear_.result().at(index);

    auto it = std::find_if(lin.coeff_.begin(), lin.coeff_.end(),
                           [this](const Scale<int64_t> &item) {
                               return HashComparator()(item.a_, var_);
                           });
    if (it != lin.coeff_.end()) {
        // TODO: k_ can be -1
        if (it->k_ != 1) {
            throw InvalidGPUVector("Vectorized non-contiguous memory "
                                   "accessis not supported");
        }

        auto toProve =
            makeEQ(makeMod(makeSub(index, var_), makeIntConst(vecLen_)),
                   makeIntConst(0));
        simplifyOnly_ = true;
        toProve = (*this)(toProve);
        simplifyOnly_ = false;
        if (!prove(toProve)) {
            throw InvalidGPUVector(FT_MSG << "Vectorized memory access should "
                                             "be aligned to the vector length: "
                                          << toProve << " does not hold");
        }
        return true;
    } else {
        return false;
    }
}

std::vector<Expr> LowerVector::getIndices(const std::vector<Expr> &indices) {
    std::vector<Expr> ret;
    ret.reserve(indices.size());
    isIndex_++;
    try {
        for (auto &&idx : indices) {
            ret.emplace_back((*this)(idx));
        }
    } catch (const InvalidGPUVector &e) {
        isIndex_--;
        throw;
    }
    isIndex_--;
    return ret;
}

Stmt LowerVector::visit(const For &op) {
    if (op->property_->vectorize_) {
        if (var_.isValid()) {
            throw InvalidGPUVector("Nested vectorized loops is not supported");
        }

        for (int vecLen : VEC_LEN) {
            var_ = makeVar(op->iter_).as<VarNode>();
            begin_ = op->begin_;
            vecLen_ = vecLen;
            For ret;
            try {
                auto toProve = makeEQ(makeMod(op->len_, makeIntConst(vecLen)),
                                      makeIntConst(0));
                simplifyOnly_ = true;
                toProve = (*this)(toProve);
                simplifyOnly_ = false;
                if (!prove(toProve)) {
                    throw InvalidGPUVector(
                        "The loop length should be divisible "
                        "by the vector length");
                }
                ret = deepCopy(op).as<ForNode>();
                ret->len_ = makeFloorDiv(ret->len_, makeIntConst(vecLen));
                ret->end_ = makeAdd(ret->begin_, ret->len_);
                ret->body_ = (*this)(ret->body_);
            } catch (const InvalidGPUVector &e) {
                WARNING(FT_MSG << "Vectorizing loop " << op->id() << "("
                               << op->metadata() << ") to length " << vecLen
                               << " failed because: " << e.what());
                var_ = nullptr;
                continue;
            }
            var_ = nullptr;
            ret->property_->vectorize_ = false; // done
            return ret;
        }
    }
    return BaseClass::visit(op);
}

Expr LowerVector::visit(const Var &op) {
    if (!simplifyOnly_ && var_.isValid() && var_->name_ == op->name_) {
        auto ret = makeAdd(makeMul(makeSub(op, begin_), makeIntConst(vecLen_)),
                           begin_);
        if (!isIndex_) {
            switch (vecLen_) {
            case 4:
                ret = makeIntrinsic("int4{%, %, %, %}",
                                    {ret, makeAdd(ret, makeIntConst(1)),
                                     makeAdd(ret, makeIntConst(2)),
                                     makeAdd(ret, makeIntConst(3))},
                                    DataType::Custom, false);
                break;
            case 2:
                ret = makeIntrinsic("int2{%, %}",
                                    {ret, makeAdd(ret, makeIntConst(1))},
                                    DataType::Custom, false);
                break;
            default:
                ASSERT(false);
            }
        }
        return ret;
    }
    return BaseClass::visit(op);
}

Expr LowerVector::visit(const Load &op) {
    if (!simplifyOnly_ && var_.isValid() && !op->indices_.empty()) {
        if (hasVectorIndices(op->indices_,
                             buffer(op->var_)->tensor()->shape())) {
            auto &&indices = getIndices(op->indices_);
            auto dtype = buffer(op->var_)->tensor()->dtype();
            auto vtype = vecType(dtype);
            return makeIntrinsic("*((" + vtype + "*)&(%))",
                                 {makeLoad(op->var_, indices, dtype)},
                                 DataType::Custom, false);
        }
    }
    return BaseClass::visit(op);
}

Stmt LowerVector::visit(const Store &op) {
    if (var_.isValid() && !op->indices_.empty()) {
        if (hasVectorIndices(op->indices_,
                             buffer(op->var_)->tensor()->shape())) {
            auto &&indices = getIndices(op->indices_);
            auto dtype = buffer(op->var_)->tensor()->dtype();
            auto vtype = vecType(dtype);
            return makeEval(makeIntrinsic(
                "*((" + vtype + "*)&(%)) = make_" + vtype + "(%)",
                {makeLoad(op->var_, indices, dtype), (*this)(op->expr_)},
                DataType::Void, false));
        }
    }
    return BaseClass::visit(op);
}

Stmt LowerVector::visit(const ReduceTo &op) {
    if (var_.isValid() && !op->indices_.empty()) {
        if (hasVectorIndices(op->indices_,
                             buffer(op->var_)->tensor()->shape())) {
            auto &&indices = getIndices(op->indices_);
            auto dtype = buffer(op->var_)->tensor()->dtype();
            auto vtype = vecType(dtype);
            auto newLoad = makeLoad(op->var_, indices, dtype);
            switch (op->op_) {
            case ReduceOp::Add:
                return makeEval(makeIntrinsic(
                    "*((" + vtype + "*)&(%)) += make_" + vtype + "(%)",
                    {newLoad, (*this)(op->expr_)}, DataType::Void, false));
            case ReduceOp::Max:
                return makeEval(
                    makeIntrinsic("*((" + vtype + "*)&(%)) = max(*((*" + vtype +
                                      ")&(%)), make_" + vtype + "(%))",
                                  {newLoad, newLoad, (*this)(op->expr_)},
                                  DataType::Void, false));
            case ReduceOp::Min:
                return makeEval(
                    makeIntrinsic("*((" + vtype + "*)&(%)) = min(*((*" + vtype +
                                      ")&(%)), make_" + vtype + "(%))",
                                  {newLoad, newLoad, (*this)(op->expr_)},
                                  DataType::Void, false));
            default:
                ASSERT(false);
            }
        }
    }
    return BaseClass::visit(op);
}

Stmt lowerVector(const Stmt &_op) {
    auto op = LowerVector()(_op);
    return simplify(op);
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
