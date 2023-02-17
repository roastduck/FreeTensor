#include <algorithm>

#include <auto_schedule/utils.h>
#include <autograd/clear_mark_version.h>
#include <codegen/code_gen.h>
#include <container_utils.h>
#include <driver.h>
#include <lower.h>
#include <omp_utils.h>
#include <pass/const_fold.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
#include <schedule.h>
#include <schedule/as_matmul.h>
#include <schedule/blend.h>
#include <schedule/cache.h>
#include <schedule/check_loop_order.h>
#include <schedule/fission.h>
#include <schedule/fuse.h>
#include <schedule/inlining.h>
#include <schedule/merge.h>
#include <schedule/multi_level_tiling.h>
#include <schedule/parallelize.h>
#include <schedule/permute.h>
#include <schedule/pluto.h>
#include <schedule/reorder.h>
#include <schedule/separate_tail.h>
#include <schedule/set_mem_type.h>
#include <schedule/split.h>
#include <schedule/swap.h>
#include <schedule/unroll.h>
#include <schedule/var_merge.h>
#include <schedule/var_reorder.h>
#include <schedule/vectorize.h>

namespace freetensor {

Stmt Schedule::quickOptimizations(const Stmt &_ast) {
    auto ast = _ast;
    ast = constFold(ast);
    ast = makeReduction(ast);
    ast = flattenStmtSeq(ast);
    return ast;
}

Schedule::Schedule(const Stmt &ast, int verbose)
    : verbose_(verbose), memoized_(Ref<MemoizedSchedules>::make()),
      rng_(Ref<OpenMPRandomEngine>::make(0)) /* TODO: set seed */,
      randCtx_(Ref<RandCtx<OpenMPRandomEngine>>::make(*rng_)) {
    openTrans_.emplace_back(quickOptimizations(clearMarkVersion(ast)),
                            ScheduleLog());
}

void Schedule::beginTransaction() { openTrans_.emplace_back(ast(), logs()); }

void Schedule::commitTransaction() {
    if (openTrans_.size() == 1) {
        ERROR(
            "The outer-most default transaction does not need to be committed");
    }
    auto trans = std::move(openTrans_.back());
    openTrans_.pop_back();
    if (verbose_ >= 2) {
        auto &&os = logger();
        auto logs = asVector(openTrans_.back().logs_, trans.logs_);
        if (!logs.empty()) {
            os << "Committing schedule(s): ";
            for (auto &&[i, item] : views::enumerate(logs)) {
                os << (i > 0 ? ", " : "") << *item;
            }
            os << ", resulting in:" << std::endl << trans.ast_ << std::endl;
        } else {
            os << "No schedule is committed" << std::endl;
        }
    }
    openTrans_.back() = std::move(trans);
}

void Schedule::abortTransaction() {
    if (openTrans_.size() == 1) {
        ERROR("The outer-most default transaction cannot be aborted");
    }
    openTrans_.pop_back();
}

const Stmt &Schedule::ast() const { return openTrans_.back().ast_; }
void Schedule::setAst(const Stmt &ast) { openTrans_.back().ast_ = ast; }

const ScheduleLog &Schedule::logs() const { return openTrans_.back().logs_; }
void Schedule::setLogs(const ScheduleLog &logs) {
    openTrans_.back().logs_ = logs;
}

void Schedule::autoSchedule(const Ref<Target> &target,
                            const Ref<RandTrace> &trace) {
    autoUseLib(target);
    autoFissionFuse(target, trace);
    autoReorder(target);
    autoParallelize(target);
    autoSetMemType(target);
    autoUnroll(target);
}

std::vector<AutoScheduleTuneTrial> Schedule::tuneAutoSchedule(
    int nBatch, int batchSize, const Ref<Device> &device,
    const std::vector<Ref<Array>> &args,
    const std::unordered_map<std::string, Ref<Array>> &kws,
    const std::regex &toLearn) {
    try {
        randCtx_->setLearn();
        randCtx_->setLearnFilter(toLearn);
        std::vector<AutoScheduleTuneTrial> trials(nBatch * batchSize);
        for (int i = 0; i < nBatch; i++) {
            if (verbose_ >= 1) {
                logger() << "Tuning auto_schedule: Batch " << i << std::endl;
            }
            std::vector<Ref<Driver>> drivers(batchSize);
            exceptSafeParallelFor<size_t>(
                0, batchSize, 1,
                [&](size_t j) {
                    auto &[trace, lowered, code, _1, _2] =
                        trials[i * batchSize + j];
                    auto s = this->fork();
                    trace = Ref<RandTrace>::make();
                    s.autoSchedule(device->target(), trace);
                    lowered = lower(s.func(), device->target());
                    code = codeGen(lowered, device->target());
                    drivers[j] = Ref<Driver>::make(lowered, code, device);
                },
                omp_sched_static); // use schedule(static) to guarantee
                                   // deterministic RNG
            for (int j = 0; j < batchSize; j++) {
                auto &d = *drivers[j];
                auto &[trace, _1, _2, t, stddev] = trials[i * batchSize + j];
                d.setArgs(args, kws);
                // TODO: Allow setting measuring repeats
                std::tie(t, stddev) = d.time();
                d.collectReturns();
                randCtx_->observeTrace(trace, t, stddev);
            }
        }
        randCtx_->setInfer();
        return trials;
    } catch (...) {
        randCtx_->setInfer();
        throw;
    }
}

std::vector<std::pair<ID, int>>
Schedule::multiLevelTiling(const ForsWithDataReuse &target,
                           const MultiLevelTilingAnnotation &annotation,
                           const std::string &pat, int level) {
    return freetensor::multiLevelTiling(*this, target, annotation, pat, level);
}
std::vector<std::pair<ID, int>> Schedule::multiLevelTilingWithFusion(
    const ForsWithDataReuse &target,
    const MultiLevelTilingAnnotation &annotation, const std::string &pat,
    const ElementWiseInfo &toFuse, int level, TargetType targetType,
    bool doCacheRead) {
    return freetensor::multiLevelTilingWithFusion(
        *this, target, annotation, pat, toFuse, level, targetType, doCacheRead);
}
} // namespace freetensor
