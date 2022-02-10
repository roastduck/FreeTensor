#include <hash.h>
#include <id.h>

namespace ir {

bool operator==(const ExprOrStmtId &lhs, const ExprOrStmtId &rhs) {
    return lhs.id_ == rhs.id_ && HashComparator()(lhs.expr_, rhs.expr_);
}

} // namespace ir

namespace std {

size_t hash<ir::ExprOrStmtId>::operator()(const ir::ExprOrStmtId &id) const {
    return ir::hashCombine(ir::Hasher()(id.expr_),
                           std::hash<std::string>()(id.id_));
}

} // namespace std

