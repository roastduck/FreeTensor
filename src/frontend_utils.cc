#include <except.h>
#include <frontend_utils.h>

namespace ir {

Expr FrontendVar::asLoad() const {
    if (indices_.size() != shape_.size()) {
        throw InvalidProgram(name_ + " is of a " +
                             std::to_string(shape_.size()) + "-D shape, but " +
                             std::to_string(indices_.size()) +
                             "-D indices are given");
    }
    std::vector<Expr> indices;
    indices.reserve(indices_.size());
    for (auto &&idx : indices_) {
        ASSERT(idx.type() == FrontendVarIdxType::Single);
        indices.emplace_back(idx.single());
    }
    return makeLoad(name_, std::move(indices));
}

Stmt FrontendVar::asStore(const std::string &id, const Expr &value) const {
    if (indices_.size() != shape_.size()) {
        throw InvalidProgram(name_ + " is of a " +
                             std::to_string(shape_.size()) + "-D shape, but " +
                             std::to_string(indices_.size()) +
                             "-D indices are given");
    }
    std::vector<Expr> indices;
    indices.reserve(indices_.size());
    for (auto &&idx : indices_) {
        ASSERT(idx.type() == FrontendVarIdxType::Single);
        indices.emplace_back(idx.single());
    }
    return makeStore(id, name_, std::move(indices), value);
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

} // namespace ir
