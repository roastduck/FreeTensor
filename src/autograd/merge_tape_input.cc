#include <analyze/all_uses.h>
#include <analyze/check_all_defined.h>
#include <analyze/find_stmt.h>
#include <autograd/merge_tape_input.h>
#include <container_utils.h>
#include <hash.h>

namespace freetensor {

Stmt MergeTapeInput::visitStmt(const Stmt &s) {
    auto ret = BaseClass::visitStmt(s);
    auto allWritesInBody = allWrites(ret);
    if (auto it = lca2newNodes_.find(s); it != lca2newNodes_.end()) {
        auto &&newNodes = it->second;
        for (auto &&newNode : newNodes) {
            for (auto &&dim : newNode->buffer_->tensor()->shape()) {
                auto allNamesInDim = allNames(dim);
                if (!checkAllDefined(names(), allNamesInDim)) {
                    ERROR(FT_MSG << "Cannot merge tape inputs, or the shape of "
                                    "the input "
                                 << newNode << " will be undefined");
                }
                if (hasIntersect(allWritesInBody, allNamesInDim)) {
                    ERROR(FT_MSG << "Cannot merge tape inputs, or the shape of "
                                    "the input "
                                 << newNode << " will be modified");
                }
            }
            ret = makeVarDef(newNode->name_, newNode->buffer_, newNode->viewOf_,
                             std::move(ret), newNode->pinned_,
                             newNode->metadata());
        }
    }
    return ret;
}

Stmt MergeTapeInput::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (namesMerging_.count(op->name_)) {
        return op->body_;
    } else {
        return op;
    }
}

Stmt mergeTapeInput(const Stmt &op) {
    std::unordered_map<std::string, std::vector<VarDef>> name2nodes;
    for (auto &&_input : findAllStmt(op, [](const Stmt &s) {
             return s->nodeType() == ASTNodeType::VarDef &&
                    (s.as<VarDefNode>()->buffer_->atype() ==
                         AccessType::Input ||
                     s.as<VarDefNode>()->buffer_->atype() ==
                         AccessType::InputMutable);
         })) {
        auto &&input = _input.as<VarDefNode>();
        name2nodes[input->name_].emplace_back(input);
    }

    std::unordered_map<Stmt, std::vector<VarDef>>
        lca2newNodes; // Hoist destination -> nodes to hoist
    std::unordered_set<std::string> namesMerging;
    for (auto &&[name, nodes] : name2nodes) {
        if (nodes.size() > 1) {
            VarDef newNode = deepCopy(nodes.front()).as<VarDefNode>();
            Stmt lca = nodes.front();
            AccessType atype = AccessType::Input;
            for (auto it = nodes.begin() + 1; it != nodes.end(); it++) {
                auto &&node = *it;
                if (!HashComparator{}(newNode->buffer_->tensor(),
                                      node->buffer_->tensor())) {
                    ERROR(FT_MSG << "Cannot merge tape input because "
                                 << newNode << " and " << node
                                 << " have different definitions");
                }
                lca = lcaStmt(lca, node);
                if (node->buffer_->atype() == AccessType::InputMutable) {
                    atype = AccessType::InputMutable;
                }
            }
            newNode->buffer_->setAtype(atype);
            lca2newNodes[lca].emplace_back(newNode);
            namesMerging.insert(name);
        }
    }

    return MergeTapeInput{lca2newNodes, namesMerging}(op);
}

} // namespace freetensor
