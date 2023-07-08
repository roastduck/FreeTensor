#include <algorithm>

#include <pass/const_fold.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoMemLayout(const Ref<Target> &target) {
    // GPU needs the variables' last dimension to have a >=32 length to map to
    // warps
    if (target->type() == TargetType::GPU) {
        for (auto &&_def : findAll("<VarDef>")) {
            auto &&def = _def.as<VarDefNode>();
            auto &&shape = def->buffer_->tensor()->shape();
            auto lastLongDimIt =
                std::find_if(shape.rbegin(), shape.rend(), [&](const Expr &d) {
                    if (auto &&l = constFold(d);
                        l->nodeType() == ASTNodeType::IntConst) {
                        return l.as<IntConstNode>()->val_ >=
                               target.as<GPUTarget>()->warpSize();
                    }
                    return true;
                });
            if (lastLongDimIt == shape.rend()) {
                continue; // All dims are short
            }
            if (lastLongDimIt == shape.rbegin()) {
                continue; // Last dim already long
            }
            auto lastLongDim =
                shape.size() - 1 - (lastLongDimIt - shape.rbegin());

            try {
                auto transformableDefId = def->id();
                if (isInputting(def->buffer_->atype()) ||
                    isOutputting(def->buffer_->atype())) {
                    auto &&[_1, _2, _3, newId] = cache(
                        def->body_->id(), def->name_, def->buffer_->mtype());
                    transformableDefId = newId;
                }

                varSplit(transformableDefId, lastLongDim,
                         VarSplitMode::RelaxedSize,
                         target.as<GPUTarget>()->warpSize());
                auto order = ranges::to<std::vector>(
                    views::ints(0, (int)shape.size() + 1));
                order.erase(order.begin() + lastLongDim + 1);
                order.emplace_back(lastLongDim + 1);
                varReorder(transformableDefId, order);
            } catch (const InvalidSchedule &e) {
                // Ignore
            }
        }
    }
}

} // namespace freetensor
