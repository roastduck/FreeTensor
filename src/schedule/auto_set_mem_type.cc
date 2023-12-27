#include <compare>
#include <map>
#include <optional>
#include <unordered_map>

#include <analyze/all_defs.h>
#include <analyze/find_stmt.h>
#include <math/utils.h>
#include <pass/const_fold.h>
#include <pass/gpu/multiplex_buffers.h>
#include <pass/gpu/simplex_buffers.h>
#include <schedule.h>

namespace freetensor {

namespace {

#ifdef FT_WITH_CUDA
std::optional<size_t> optMul(const std::optional<size_t> &lhs,
                             const std::optional<size_t> &rhs) {
    if (lhs.has_value() && rhs.has_value()) {
        return *lhs * *rhs;
    } else {
        return std::nullopt;
    }
}

struct SizeOnEachLevel {
    size_t dtypeSize_;
    // Original size of the VarDef node
    std::optional<size_t> defSize_;
    // Sizes on each parallel level
    std::optional<size_t> thread_, block_, grid_;
};

inline auto operator==(const SizeOnEachLevel &lhs, const SizeOnEachLevel &rhs) {
    return lhs.defSize_ == rhs.defSize_;
}

/**
 * SizeOnEachLevel can be sorted by defSize_, no matter size of which level or
 * which restriction we care for, because threadDim and gridDim is fixed for
 * each kernel.
 *
 * NOTE: Some restricions count in bytes, while others count in number of words,
 * which may lead to some different orders when sorting by different criteria.
 * We ignore such a difference by now.
 *
 * Unknown sizes are treated as largest
 */
inline std::strong_ordering operator<=>(const SizeOnEachLevel &lhs,
                                        const SizeOnEachLevel &rhs) {
    if (lhs.defSize_.has_value() && rhs.defSize_.has_value()) {
        return *lhs.defSize_ <=> *rhs.defSize_;
    } else if (!lhs.defSize_.has_value() && rhs.defSize_.has_value()) {
        return std::strong_ordering::greater;
    } else if (lhs.defSize_.has_value() && !rhs.defSize_.has_value()) {
        return std::strong_ordering::less;
    } else {
        return std::strong_ordering::equal;
    }
}

/**
 * The size of each variable PER THREAD BLOCK is known only after
 * pass/gpu/multiplex_buffers and pass/gpu/simplex_buffers. We try setting the
 * variable to GPUShared first, dry-run the two passes to get the actual size,
 * and then reset the schedule.
 *
 * Resources used by parallel reduction is not considered for now.
 */
SizeOnEachLevel estimateSizeOnEachLevel(Schedule &s, const ID &defId,
                                        MemType mtype,
                                        const Ref<GPUTarget> &target) {
    SizeOnEachLevel ret;
    s.beginTransaction();
    try {
        s.setMemType(defId, mtype);
        auto ast = s.ast();
        ast = gpu::multiplexBuffers(ast, target, defId);
        ast = gpu::simplexBuffers(ast, defId);
        // No normalizeThreads here because it is too slow. If only a master
        // thread is using gpu/local while other threads is not, we may
        // under-estimate the size, since idle threads are also holding
        // registers. We ignore this case for now.
        ast = constFold(ast); // for lengths
        Stmt _newVarDef;
        try {
            _newVarDef = findStmt(ast, defId);
        } catch (const UnexpectedQueryResult &e) {
            // Maybe a trivial VarDef that can be optimized out
            s.abortTransaction();
            return ret;
        }
        ASSERT(_newVarDef->nodeType() == ASTNodeType::VarDef);
        auto newVarDef = _newVarDef.as<VarDefNode>();

        ret.dtypeSize_ = sizeOf(newVarDef->buffer_->tensor()->dtype());
        ret.defSize_ = ret.dtypeSize_;
        for (auto &&dim : newVarDef->buffer_->tensor()->shape()) {
            if (dim->nodeType() == ASTNodeType::IntConst) {
                *ret.defSize_ *= dim.as<IntConstNode>()->val_;
            } else {
                ret.defSize_ = std::nullopt;
                break;
            }
        }

        std::optional<size_t> blockDim = 1, gridDim = 1;
        for (Stmt p = newVarDef; p.isValid(); p = p->parentStmt()) {
            if (p->nodeType() == ASTNodeType::For) {
                auto &&loop = p.as<ForNode>();
                std::optional<size_t> len;
                if (loop->len_->nodeType() == ASTNodeType::IntConst) {
                    len = loop->len_.as<IntConstNode>()->val_;
                }
                if (std::holds_alternative<CUDAScope>(
                        loop->property_->parallel_)) {
                    switch (std::get<CUDAScope>(loop->property_->parallel_)
                                .level_) {
                    case CUDAScope::Thread:
                        blockDim = optMul(blockDim, len);
                        break;
                    case CUDAScope::Block:
                        gridDim = optMul(gridDim, len);
                        break;
                    default:
                        ASSERT(false);
                    }
                }
            }
        }

        switch (mtype) {
        case MemType::GPULocal:
            ret.thread_ = ret.defSize_;
            ret.block_ = optMul(ret.thread_, blockDim);
            ret.grid_ = optMul(ret.block_, gridDim);
            break;
        case MemType::GPUWarp:
        case MemType::GPUShared:
            ret.block_ = ret.defSize_;
            ret.grid_ = optMul(ret.block_, gridDim);
            break;
        case MemType::GPUGlobal:
        case MemType::GPUGlobalHeap:
            ret.grid_ = ret.defSize_;
            break;
        default:
            ASSERT(false);
        }
    } catch (const InvalidSchedule &e) {
        // do nothing
    }
    s.abortTransaction();
    return ret;
}

bool maybeKernelBoundary(const Stmt &s) {
    if (s->nodeType() == ASTNodeType::For &&
        std::holds_alternative<CUDAScope>(
            s.as<ForNode>()->property_->parallel_)) {
        return true;
    } else if (s->nodeType() == ASTNodeType::VarDef) {
        auto mtype = s.as<VarDefNode>()->buffer_->mtype();
        if (mtype == MemType::GPULocal || mtype == MemType::GPUWarp ||
            mtype == MemType::GPUShared) {
            return true;
        }
    }
    return false;
}

Stmt findKernelBoundaryOutwards(const Stmt &s) {
    return s->parentStmtByFilter(maybeKernelBoundary);
}

std::vector<Stmt> findKernelBoundariesInwards(const Stmt &s) {
    class Finder : public Visitor {
        std::vector<Stmt> found_;

      public:
        const auto &found() const { return found_; }

      protected:
        void visitStmt(const Stmt &s) override {
            if (maybeKernelBoundary(s)) {
                found_.emplace_back(s);
                // no recurse
            } else {
                Visitor::visitStmt(s);
            }
        }
    };

    Finder finder;
    finder(s);
    return finder.found();
}

/**
 * New kernel boundary after setting mtype for `s`
 */
Stmt findNewKernelNode(const Stmt &s) {
    Stmt ret = s; // The root of a kernel can be a shared memory VarDef
    for (auto p = s->parentStmt(); p.isValid(); p = p->parentStmt()) {
        if (maybeKernelBoundary(p)) {
            ret = p;
        }
    }
    return ret;
};
#endif // FT_WITH_CUDA

} // Anonymous namespace

void Schedule::autoSetMemType(const Ref<Target> &target) {
    // Try to put each VarDef as near to processor as possible

#ifdef FT_WITH_CUDA
    if (target->type() == TargetType::GPU) {
        // All variables are in GPUGlobal in the first place. First try to user
        // GPULocal, if failed, then GPUShared.
        //
        // Since the restrictions on each memory type depends on the total sizes
        // of all variables of the type, instead of each individual variables,
        // we sort variables by their sizes, and try to set its memory type from
        // smallers ones to larger ones, until we reach the limit.
        // Dyanmic-shaped varialbes are treated as largest. Restrictions on each
        // memory types are listed as follows:
        //
        // To use GPULocal, the following restrictions must be satisfied:
        //
        // a) Shapes per thread block shall not be larger than the register
        // count. Even we put them to local memory, they will still land on
        // DRAM. Since NVCC allocates stack frames (where our local memory
        // variables go to,
        // https://forums.developer.nvidia.com/t/out-of-memory-when-allocating-local-memory/238615)
        // by maximum possible thread count, so it will be worse than using
        // global memory.
        // b) Shapes per thread shall not be larger than the maximum stack-frame
        // size divided by maximum possble thread count, for the reason above.
        // Since we have no idea whether NVCC will put a local variable in local
        // memory or registers, in case NVCC does put it in local memory, we
        // don't want an OOM.
        //
        // To use GPUShared, we should not overrun the size limit per thread
        // block.

        auto allDefsVec = allDefs(ast(), {AccessType::Cache});
        // We only set memory types for variables used in only one kernel.
        // Setting memory types for cross-kernel variables will implicitly merge
        // the kernels, which may lead to illegal dependences currently
        // unchecked in the `set_mem_type` schedule. For example, setting `t` to
        // be `gpu/shared` in
        //
        // ```
        // @!parallel blockIdx.x for i = ...  { t[i] = ...; }
        // @!parallel blockIdx.x for i = ... { ... = t[i + 1]; }
        // ```
        //
        // results in a cross-block dependence.
        auto all = ranges::to<std::unordered_map>(
            allDefsVec | views::filter([this](auto &&pair) {
                auto defNode = find(pair.first);
                if (findKernelBoundaryOutwards(defNode).isValid()) {
                    return true; // Inside a kernel
                }
                if (findKernelBoundariesInwards(defNode).size() <= 1) {
                    return true; // Out of no more than 1 kernel
                }
                return false;
            }));
        std::multimap<SizeOnEachLevel, ID> sortedSize2defId; // small to large

        // Try setting to GPULocal
        for (auto &&[defId, name] : all) {
            sortedSize2defId.emplace(
                estimateSizeOnEachLevel(*this, defId, MemType::GPULocal,
                                        target.as<GPUTarget>()),
                defId);
        }
        std::unordered_map<ID, size_t> kernelLocalSizePerThread_,
            kernelLocalSizePerBlock_;
        size_t localSizeLimPerThread =
            target.as<GPUTarget>()->maxLocalMemorySizePerThread();
        size_t localSizeLimPerBlock = target.as<GPUTarget>()->regsPerBlock();
        for (auto &&[size, defId] : sortedSize2defId) {
            auto kernel = findNewKernelNode(find(defId))->id();
            auto regPerWord = ceilDiv(size.dtypeSize_, 4); // 32-bit register
            if (size.block_.has_value() &&
                kernelLocalSizePerBlock_[kernel] + *size.block_ * regPerWord <
                    localSizeLimPerBlock) {
                if (size.block_.has_value() &&
                    kernelLocalSizePerThread_[kernel] + *size.thread_ <
                        localSizeLimPerThread) {
                    try {
                        setMemType(defId, MemType::GPULocal);
                        kernelLocalSizePerThread_[kernel] += *size.thread_;
                        kernelLocalSizePerBlock_[kernel] +=
                            *size.block_ * regPerWord;
                        all.erase(defId);
                    } catch (const InvalidSchedule &e) {
                        // do nothing
                    }
                }
            }
        }
        sortedSize2defId.clear();

        // Try setting to GPUShared
        for (auto &&[defId, name] : all) {
            sortedSize2defId.emplace(
                estimateSizeOnEachLevel(*this, defId, MemType::GPUShared,
                                        target.as<GPUTarget>()),
                defId);
        }
        std::unordered_map<ID, size_t> kernelSharedSize_;
        auto sharedSizeLim = target.as<GPUTarget>()->sharedMemPerBlock();
        for (auto &&[size, defId] : sortedSize2defId) {
            auto kernel = findNewKernelNode(find(defId))->id();
            if (size.block_.has_value() &&
                kernelSharedSize_[kernel] + *size.block_ < sharedSizeLim) {
                try {
                    setMemType(defId, MemType::GPUShared);
                    kernelSharedSize_[kernel] += *size.block_;
                    all.erase(defId);
                } catch (const InvalidSchedule &e) {
                    // do nothing
                }
            }
        }
        sortedSize2defId.clear();
    }
#endif // FT_WITH_CUDA
}

} // namespace freetensor
