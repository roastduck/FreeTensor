#include <analyze/find_indexing_loops.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoUnroll(const Target &target) {
    if (target.type() == TargetType::GPU) {
        // Try to unroll loops that accessing local arrays, to help nvcc put
        // these arrays to registers
        for (auto &&[loop, defs] : findIndexingLoops(ast())) {
            if (loop->property_->parallel_ != serialScope ||
                loop->property_->vectorize_) {
                continue;
            }

            for (auto &&def : defs) {
                if (def->buffer_->mtype() == MemType::GPULocal) {
                    goto do_unroll;
                }
            }
            continue;
        do_unroll:
            try {
                unroll(loop->id());
            } catch (InvalidSchedule &e) {
                // do nothing
            }
        }
    }

    // Unroll very short loops
    for (auto &&_loop : findAll("<For>")) {
        auto loop = _loop.as<ForNode>();
        if (loop->property_->parallel_ == serialScope &&
            !loop->property_->vectorize_ && !loop->property_->unroll_ &&
            loop->len_->nodeType() == ASTNodeType::IntConst &&
            loop->len_.as<IntConstNode>()->val_ <= 4) {
            unroll(loop->id());
        }
    }
}

} // namespace freetensor
