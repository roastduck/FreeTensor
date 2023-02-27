#include <analyze/all_uses.h>
#include <container_utils.h>
#include <except.h>
#include <frontend/frontend_var.h>

namespace freetensor {

int FrontendVar::ndim() const {
    int ndim = fullShape_.size();
    for (auto &&idx : indices_) {
        if (idx.type() == FrontendVarIdxType::Single) {
            ndim--;
        }
    }
    return ndim;
}

Expr FrontendVar::shape(const Expr &idx) const {
    Expr ret;
    size_t k = 0;
    for (size_t i = 0, n = fullShape_.size(); i < n; i++) {
        if (i < indices_.size() &&
            indices_[i].type() == FrontendVarIdxType::Single) {
            continue;
        }
        Expr dimLen = fullShape_.at(i);
        if (i < indices_.size() &&
            indices_[i].type() == FrontendVarIdxType::Slice) {
            dimLen = indices_[i].len();
        }
        if (idx->nodeType() == ASTNodeType::IntConst &&
            (size_t)idx.as<IntConstNode>()->val_ == k) {
            return dimLen;
        } else {
            ret = ret.isValid()
                      ? makeIfExpr(makeEQ(idx, makeIntConst(k)), dimLen, ret)
                      : dimLen;
        }
        k++;
    }
    ASSERT(ret.isValid());
    return ret;
}

std::vector<Expr> FrontendVar::shape() const {
    std::vector<Expr> ret;
    for (size_t i = 0, n = fullShape_.size(); i < n; i++) {
        if (i < indices_.size() &&
            indices_[i].type() == FrontendVarIdxType::Single) {
            continue;
        }
        Expr dimLen = fullShape_.at(i);
        if (i < indices_.size() &&
            indices_[i].type() == FrontendVarIdxType::Slice) {
            dimLen = indices_[i].len();
        }
        ret.emplace_back(dimLen);
    }
    return ret;
}

Expr FrontendVar::asLoad() const {
    if (ndim() != 0) {
        throw InvalidProgram(
            name_ + " is of a " + std::to_string(fullShape_.size()) +
            "-D shape, but " + std::to_string((int)fullShape_.size() - ndim()) +
            "-D indices are given");
    }
    std::vector<Expr> indices;
    indices.reserve(indices_.size());
    for (auto &&idx : indices_) {
        ASSERT(idx.type() == FrontendVarIdxType::Single);
        indices.emplace_back(idx.single());
    }
    if (!isLoadAtVersion_) {
        return makeLoad(name_, std::move(indices), dtype_);
    } else {
        return _makeLoadAtVersion(name_, std::move(indices), dtype_);
    }
}

Stmt FrontendVar::asStore(const Metadata &metadata, const Expr &value) const {
    if (ndim() != 0) {
        throw InvalidProgram(
            name_ + " is of a " + std::to_string(fullShape_.size()) +
            "-D shape, but " + std::to_string((int)fullShape_.size() - ndim()) +
            "-D indices are given");
    }
    if (isLoadAtVersion_) {
        throw InvalidAutoGrad(
            "Unable to assign to a forward variable (a `mark_version` "
            "variable) from a `UserGrad` scope");
    }
    std::vector<Expr> indices;
    indices.reserve(indices_.size());
    for (auto &&idx : indices_) {
        ASSERT(idx.type() == FrontendVarIdxType::Single);
        indices.emplace_back(idx.single());
    }
    return makeStore(name_, std::move(indices), value, metadata);
}

Stmt FrontendVar::asReduceTo(ReduceOp op, const Metadata &metadata,
                             const Expr &value, bool atomic) const {
    if (ndim() != 0) {
        throw InvalidProgram(
            name_ + " is of a " + std::to_string(fullShape_.size()) +
            "-D shape, but " + std::to_string((int)fullShape_.size() - ndim()) +
            "-D indices are given");
    }
    std::vector<Expr> indices;
    indices.reserve(indices_.size());
    for (auto &&idx : indices_) {
        ASSERT(idx.type() == FrontendVarIdxType::Single);
        indices.emplace_back(idx.single());
    }
    return makeReduceTo(name_, std::move(indices), op, value, atomic, metadata);
}

std::vector<FrontendVarIdx>
FrontendVar::chainIndices(const std::vector<FrontendVarIdx> &next) const {
    std::vector<FrontendVarIdx> indices;
    auto it = next.begin();
    for (auto &&idx : indices_) {
        if (idx.type() == FrontendVarIdxType::Single) {
            indices.emplace_back(idx);
        } else {
            if (it != next.end()) {
                if (it->type() == FrontendVarIdxType::Single) {
                    indices.emplace_back(FrontendVarIdx::fromSingle(
                        makeAdd(idx.start(), it->single())));
                } else {
                    indices.emplace_back(FrontendVarIdx::fromSlice(
                        makeAdd(idx.start(), it->start()),
                        makeAdd(idx.stop(), it->start())));
                }
                it++;
            } else {
                indices.emplace_back(idx); // still a slice
            }
        }
    }
    for (; it != next.end(); it++) {
        indices.emplace_back(*it);
    }
    return indices;
}

std::unordered_set<std::string> allReads(const FrontendVarIdx &idx) {
    switch (idx.type()) {
    case FrontendVarIdxType::Single:
        return allReads(idx.single());
    case FrontendVarIdxType::Slice:
        return uni(allReads(idx.start()), allReads(idx.stop()));
    default:
        ASSERT(false);
    }
}

} // namespace freetensor
