#include <pass/const_fold.h>
#include <schedule.h>
#include <visitor.h>

namespace freetensor {

namespace {

class GetDFSPostOrderOfVarDef : public Visitor {
    std::vector<VarDef> order_;

  public:
    const auto &order() const { return order_; }

  protected:
    void visit(const VarDef &op) override {
        Visitor::visit(op);
        order_.emplace_back(op);
    }
};

std::vector<VarDef> getDFSPostOrderOfVarDef(const Stmt &ast) {
    GetDFSPostOrderOfVarDef visitor;
    visitor(ast);
    return visitor.order();
}

} // namespace

void Schedule::autoInline(const Ref<Target> &target) {
    // Inline very-small VarDef nodes
    //
    // Since we don't support inlining a VarDef resulted from reduction, the
    // cost to compute a VarDef is proportional to its size. Thus small VarDef
    // nodes are easy to compute, so they can be inlined

    for (auto &&def : getDFSPostOrderOfVarDef(ast())) {
        auto size = makeIntConst(1);
        for (auto &&dim : def->buffer_->tensor()->shape()) {
            size = makeMul(size, dim);
        }
        if (auto s = constFold(size); s->nodeType() == ASTNodeType::IntConst) {
            if (s.as<IntConstNode>()->val_ <= 32) {
                try {
                    inlining(def->id());
                } catch (const InvalidSchedule &e) {
                    // Do nothing
                }
            }
        }
    }
}

} // namespace freetensor
