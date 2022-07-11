#include <analyze/get_loop_nest_tree.h>
#include <auto_schedule/rules/parallelize.h>
#include <auto_schedule/rules/thread_bind.h>
#include <auto_schedule/rules/unroll.h>
#include <auto_schedule/utils.h>
#include <schedule/unroll.h>

namespace freetensor {

static std::vector<int> unrollConfigsCpu = {0, 16, 64, 512};
static std::vector<int> unrollConfigsGpu = {0, 16, 64, 512, 1024};

void UnrollPart::apply(Schedule &schedule, SubSketch &subSketch) {
    Stmt root;
    int vthreadSize = 1;
    if (targetType_ == TargetType::GPU) {
        SketchPart part = subSketch.getPart(SketchPartType::ThreadBind);
        ID lastParallelizedID = part.as<ThreadBindPart>()->lastParallelizedID();
        if (!lastParallelizedID.isValid()) {
            return;
        }
        root = schedule.find(lastParallelizedID).as<ForNode>()->body_;
        vthreadSize = part.as<ThreadBindPart>()->vthreadSize();
    } else {
        SketchPart part = subSketch.getPart(SketchPartType::Parallelize);
        ID lastParallelizedID =
            part.as<ParallelizePart>()->lastParallelizedID();
        if (!lastParallelizedID.isValid()) {
            return;
        }
        root = schedule.find(lastParallelizedID).as<ForNode>()->body_;
    }
    std::function<int(const Ref<LoopNest> &nest)> visitNest =
        [&](const Ref<LoopNest> &nest) {
            int sz = 0;
            for (auto &&subNest : nest->subLoops_) {
                sz += visitNest(subNest);
            }
            if (sz == 0) {
                sz = vthreadSize;
            }
            auto &&loop = nest->loop_;
            if (loop.isValid()) { // not root
                if (loop->property_->parallel_ == serialScope &&
                    !loop->property_->vectorize_ && !loop->property_->unroll_ &&
                    loop->len_->nodeType() == ASTNodeType::IntConst &&
                    sz * loop->len_.as<IntConstNode>()->val_ <= maxSize_) {
                    sz *= loop->len_.as<IntConstNode>()->val_;
                    schedule.unroll(loop->id());
                }
            }
            return sz;
        };
    visitNest(getLoopNestTree(root));
}

void UnrollPart::genRandAnnotation(RNG &gen) {
    std::vector<int> &unrollConfigs =
        targetType_ == TargetType::GPU ? unrollConfigsGpu : unrollConfigsCpu;
    maxSize_ = unrollConfigs[randomInt(unrollConfigs.size() - 1, gen)];
}

void UnrollPart::genFakeAnnotation(RNG &gen) { maxSize_ = 16; }

bool UnrollPart::mutate(RNG &gen) {
    std::vector<int> &unrollConfigs =
        targetType_ == TargetType::GPU ? unrollConfigsGpu : unrollConfigsCpu;
    maxSize_ = unrollConfigs[randomInt(unrollConfigs.size() - 1, gen)];
    return true;
}
bool UnrollPart::crossover(const SketchPart &part, RNG &gen) {
    if (auto p = part.as<UnrollPart>(); p.isValid()) {
        maxSize_ = p->maxSize_;
        return true;
    }
    return false;
}

std::vector<Ref<Sketch>> UnrollRule::genPart(const Sketch &sketch) {
    auto newSketch = sketch.clone();
    newSketch->addPart(Ref<UnrollPart>::make(targetType_));
    newSketch->addLog("unroll");
    return {newSketch};
}

RuleStatus UnrollRule::analyze(const Sketch &sketch) {
    if (sketch.nowSubSketch().hasPart(SketchPartType::Unroll))
        return RuleStatus::Skip;
    if (sketch.nowSubSketch().hasPart(
            SketchPartType::MultiLevelTilingWithFusion) ||
        sketch.nowSubSketch().hasPart(
            SketchPartType::MultiLevelTilingWithFusion))
        return RuleStatus::ApplyAndSkipRest;
    return RuleStatus::Skip;
}

} // namespace freetensor
