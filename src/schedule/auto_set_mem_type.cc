#include <analyze/all_defs.h>
#include <analyze/find_stmt.h>
#include <pass/gpu/multiplex_buffers.h>
#include <pass/gpu/simplex_buffers.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoSetMemType(const Ref<Target> &target) {
    // Try to put each VarDef as near to processor as possible

#ifdef FT_WITH_CUDA
    if (target->type() == TargetType::GPU) {
        // All variables are in GPUGlobal in the first place. First try to user
        // GPULocal, if failed, then GPUShared.
        //
        // To use GPUShared, we should not overrun the size limit. (TODO: also
        // compute the size for GPULocal) However, the size of each variable PER
        // THREAD BLOCK is known only after pass/gpu/multiplex_buffers and
        // pass/gpu/simplex_buffers. We use the following approach: (TODO: also
        // consider shared memory usage from parallel reduction)
        //
        // 1. Set each variable to GPUShared one by one, and dry-run the two
        // passes to get the actual size, and then reset the schedule.
        // 2. Sort variables by their sizes, try to set them to GPUShared from
        // smallers ones to larger ones, until we reach the limit.
        // Dyanmic-shaped varialbes are treated as largest.

        auto all = allDefs(ast(), {AccessType::Cache});

        // Try setting to GPULocal
        std::unordered_set<ID> setToLocal;
        for (auto &&[defId, name] : all) {
            try {
                setMemType(defId, MemType::GPULocal);
                setToLocal.insert(defId);
            } catch (const InvalidSchedule &e) {
                // do nothing
            }
        }

        // Dry-run to get actual sizes of GPUShared. UINT_MAX = dynamic
        std::vector<std::pair<ID, size_t>> sharedSizes;
        for (auto &&[defId, name] : all) {
            if (!setToLocal.count(defId)) {
                beginTransaction();
                try {
                    setMemType(defId, MemType::GPUShared);
                    auto ast = this->ast();
                    ast = gpu::multiplexBuffers(ast, target.as<GPUTarget>(),
                                                defId);
                    ast = gpu::simplexBuffers(ast, defId);
                    auto _newVarDef = findStmt(ast, defId);
                    ASSERT(_newVarDef->nodeType() == ASTNodeType::VarDef);
                    auto newVarDef = _newVarDef.as<VarDefNode>();
                    size_t size = 1;
                    for (auto &&dim : newVarDef->buffer_->tensor()->shape()) {
                        if (dim->nodeType() == ASTNodeType::IntConst) {
                            size *= dim.as<IntConstNode>()->val_;
                        } else {
                            size = UINT_MAX;
                            break;
                        }
                    }
                    sharedSizes.emplace_back(defId, size);
                } catch (const InvalidSchedule &e) {
                    // do nothing
                }
                abortTransaction();
            }
        }
        std::sort(sharedSizes.begin(), sharedSizes.end(),
                  [](const std::pair<ID, size_t> &lhs,
                     const std::pair<ID, size_t> &rhs) {
                      return lhs.second < rhs.second;
                  });

        // Try setting to GPUShared
        std::unordered_map<ID, size_t> kernelTotalSize_;
        auto findKernelNode = [](const Stmt &_s) {
            Stmt ret = _s; // The root of a kernel can be a shared memory VarDef
            for (auto s = _s->parentStmt(); s.isValid(); s = s->parentStmt()) {
                if (s->nodeType() == ASTNodeType::For &&
                    std::holds_alternative<CUDAScope>(
                        s.as<ForNode>()->property_->parallel_)) {
                    ret = s;
                }
            }
            return ret;
        };
        auto sharedSizeLim = target.as<GPUTarget>()->sharedMemPerBlock();
        for (auto &&[defId, size] : sharedSizes) {
            auto kernel = findKernelNode(find(defId))->id();
            if (size != UINT_MAX &&
                kernelTotalSize_[kernel] + size < sharedSizeLim) {
                try {
                    setMemType(defId, MemType::GPUShared);
                    kernelTotalSize_[kernel] += size;
                } catch (const InvalidSchedule &e) {
                    // do nothing
                }
            }
        }
    }
#endif // FT_WITH_CUDA
}

} // namespace freetensor
