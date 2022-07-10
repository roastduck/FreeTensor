#include <cmath>
#include <queue>
#include <utility>

#include <analyze/find_elementwise.h>
#include <analyze/fixed_length_feature.h>
#include <analyze/structural_feature.h>
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/cache_write.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling_with_fusion.h>
#include <auto_schedule/rules/parallelize.h>
#include <auto_schedule/rules/skip.h>
#include <auto_schedule/rules/thread_bind.h>
#include <auto_schedule/rules/unroll.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>
#include <omp_utils.h>

namespace freetensor {

AutoSchedule::AutoSchedule(
    const Schedule &schedule, const Ref<Target> &target,
    const Ref<Device> &device,
    const std::function<Predicts(const Features &)> &predictFunc,
    const std::function<void(const Features &, const Predicts &)> &updateFunc,
    std::string tag, int minBlockSize,
    const std::optional<std::unordered_set<std::string>> &ruleSet, int verbose)
    : original_(schedule.clone()), target_(target), device_(device),
      paramsSet_(false), predictFunc_(std::move(predictFunc)),
      updateFunc_(std::move(updateFunc)), tag_(std::move(tag)),
      minBlockSize_(minBlockSize), verbose_(verbose) {
    flop_ = 0;
    auto opCnt =
        structuralFeature(original_.ast())[original_.ast()->id()].opCnt_;
    for (auto cnt : opCnt) {
        flop_ += cnt.second;
    }

#define ADD_RULE(name, RuleType, ...)                                          \
    if (!ruleSet || ruleSet->count(name)) {                                    \
        rules_.emplace_back(name, Ref<RuleType>::make(__VA_ARGS__));           \
    }

    if (target->type() == TargetType::CPU) {
        ADD_RULE("cache_write", CacheWriteRule, target->type(), verbose_);
        ADD_RULE("multi_level_tiling_with_fusion",
                 MultiLevelTilingWithFusionRule, target->type());
        ADD_RULE("multi_level_tiling", MultiLevelTilingRule, target->type());
        ADD_RULE("parallelize", ParallelizeRule);
        ADD_RULE("unroll", UnrollRule, target->type());
    } else {
        ADD_RULE("cache_write", CacheWriteRule, target->type(), verbose_);
        ADD_RULE("multi_level_tiling_with_fusion",
                 MultiLevelTilingWithFusionRule, target->type(), minBlockSize);
        ADD_RULE("thread_bind", ThreadBindRule);
        ADD_RULE("unroll", UnrollRule, target->type());
    }

#undef ADD_RULE

    rules_.emplace_back("skip", Ref<SkipRule>::make());
    std::random_device rd;
    randGen_ = std::default_random_engine(rd());
}

void AutoSchedule::setParams(
    const std::vector<Ref<Array>> &args,
    const std::unordered_map<std::string, Ref<Array>> &kws) {
    args_ = args;
    kws_ = kws;
    paramsSet_ = true;
}

std::vector<double> AutoSchedule::measure(std::vector<Ref<Sketch>> &sketches) {
    // Compile in parallel, and measure sequentially
    // TODO: Parallel among computing nodes

    size_t n = sketches.size();
    std::vector<Ref<Driver>> drivers(n);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            drivers[i] = Ref<Driver>::make(sketches[i]->lowered(),
                                           sketches[i]->code(), device_);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR measure: " << e.what() << std::endl;
            drivers[i] = nullptr;
        }
    }

    std::vector<double> times;
    times.reserve(n);
    for (size_t i = 0; i < n; i++) {
        ASSERT(paramsSet_);
        try {
            if (!drivers[i].isValid()) {
                times.emplace_back(1e30);
                continue;
            }
            drivers[i]->setArgs(args_, kws_);
            double t = drivers[i]->time(5, 20);
            times.emplace_back(t);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR measure: " << e.what() << std::endl;
            std::cerr << toString(sketches[i]->code()) << std::endl;
            times.emplace_back(1e30);
        }
    }
    return times;
}

void AutoSchedule::searchOneRound(size_t n, size_t nExploit, size_t nExplore) {
    ASSERT(n == nExploit + nExplore);
    if (baseSketches_.empty()) { // first time
        genSketches();
        testAndAdd(getRandPopulation(n));
    } else {
        testAndAdd(evolutionarySearch(nExploit));
        testAndAdd(getRandPopulation(nExplore));
    }
    auto bs = getBestSchedule();
    if (verbose_ >= 1) {
        logger() << "Best schedule:" << std::endl;
        for (auto log : bs.logs()) {
            logger() << log << std::endl;
        }
        logger() << "Best AST: " << std::endl
                 << toString(measuredSketches_[0]->genSchedule().ast())
                 << std::endl;
    }
}

std::vector<std::vector<double>>
AutoSchedule::genFeatures(std::vector<Ref<Sketch>> &sketches) {
    exceptSafeParallelFor<size_t>(
        0, sketches.size(), 1,
        [&](size_t i) { sketches[i]->genFeature(target_); }, omp_sched_dynamic);
    std::vector<std::vector<double>> featureList;
    for (auto &i : sketches) {
        featureList.emplace_back(i->feature());
    }
    return featureList;
}

std::vector<double>
AutoSchedule::testAndAdd(const std::vector<Ref<Sketch>> &sketches_in) {
    if (verbose_ >= 1) {
        logger() << "Lowering code" << std::endl;
    }
    std::vector<Ref<Sketch>> sketches;
    size_t nIn = sketches_in.size();
#pragma omp parallel for
    for (size_t i = 0; i < nIn; i++) {
        try {
            sketches_in[i]->genCode(target_);
        } catch (const std::exception &e) {
        }
    }
    for (size_t i = 0; i < nIn; i++) {
        if (!sketches_in[i]->code().empty()) {
            sketches.push_back(sketches_in[i]);
        }
    }
    if (verbose_ >= 1) {
        logger() << "Generating features" << std::endl;
    }
    auto features = genFeatures(sketches);
    size_t n = sketches.size();
    ASSERT(sketches.size() == n);
    std::vector<double> times = measure(sketches);
    std::vector<double> flopsList;
    for (size_t i = 0; i < times.size(); i++) {
        if (times[i] > 1e20) {
            continue;
        }
        flopsList.emplace_back(flop_ / times[i]);
    }
    updateFunc_(features, flopsList);
    double avg = 0;
    int cnt = 0;
    for (size_t i = 0; i < n; i++) {
        if (times[i] > 1e20) {
            continue;
        }
        cnt++;
        avg += times[i];
        measuredSketches_.emplace_back(sketches[i]);
        measuredSketches_.back()->setTime(times[i]);
        measuredHashes_.insert(sketches[i]->hash());
    }
    avg /= cnt;
    std::sort(times.begin(), times.end());
    std::sort(measuredSketches_.begin(), measuredSketches_.end(),
              [](const auto &a, const auto &b) { return *a < *b; });
    if (verbose_ >= 1) {
        logger() << "global min: " << measuredSketches_.front()->time()
                 << std::endl;
        logger() << "this round: min: " << times[0] << " avg: " << avg
                 << " mid: " << times[(times.size() - 1) / 2] << std::endl;
    }
    return times;
}

Schedule AutoSchedule::getBestSchedule() {
    if (measuredSketches_.empty()) {
        return {};
    }
    return measuredSketches_[0]->genSchedule();
}

double AutoSchedule::getBestTime() {
    if (measuredSketches_.empty()) {
        return 1e30;
    }
    return measuredSketches_[0]->time();
}

std::vector<Ref<Sketch>> AutoSchedule::getRandPopulation(size_t nRand) {
    std::vector<Ref<Sketch>> ret;
    std::set<size_t> used(measuredHashes_);
    std::vector<std::default_random_engine> gens;
    for (size_t i = 0; i < nRand; i++) {
        gens.emplace_back((i + i) * randGen_());
    }
    int roundUnchanged = 0;
    while (ret.size() < nRand) {
        std::vector<Ref<Sketch>> now(nRand);
        size_t nThisTurn = nRand;
#pragma omp parallel for
        for (size_t i = 0; i < nThisTurn; i++) {
            now[i] = Ref<Sketch>::make(
                baseSketches_[randomInt(baseSketches_.size() - 1, gens[i])]
                    ->genRandAnnotation(gens[i]));
            try {
                now[i]->genCode(target_);
            } catch (const std::exception &e) {
                now[i] = nullptr;
            };
        }
        roundUnchanged++;
        for (size_t i = 0; i < nThisTurn; i++) {
            if (!now[i].isValid() || now[i]->code().empty()) {
                continue;
            }
            size_t h = now[i]->hash();
            if (!used.count(h)) {
                used.insert(h);
                ret.push_back(now[i]);
                roundUnchanged = 0;
            }
            if (ret.size() >= nRand) {
                break;
            }
        }
        if (roundUnchanged > 10) {
            break;
        }
    }
    ASSERT(ret.size() == nRand);
    return ret;
}

std::vector<Ref<Sketch>> AutoSchedule::evolutionarySearch(size_t outSize) {
    // Meta-parameters used for evolutionary search. Population in an
    // evolutionary search is different from the global population
    constexpr size_t EVOLUTIONARY_SEARCH_POPULATION = 512;
    constexpr size_t EVOLUTIONARY_SEARCH_INIT_EXPLORE_CNT = 358;
    constexpr size_t EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT = 128;
    constexpr int EVOLUTIONARY_SEARCH_ITERS = 4;
    constexpr double EVOLUTIONARY_SEARCH_MUTATION_PROB = 0.6;
    constexpr double EVOLUTIONARY_SEARCH_CROSSOVER_PROB = 0.3;

    if (verbose_ >= 1) {
        logger() << "Evolutionary search" << std::endl;
    }

    std::vector<Ref<Sketch>> init =
        getRandPopulation(EVOLUTIONARY_SEARCH_INIT_EXPLORE_CNT);
    if (measuredSketches_.size() > EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT) {
        ASSERT(EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT > 0);
        measuredSketches_.resize(EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT);
    }
    for (size_t i = 0; i < EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT &&
                       i < measuredSketches_.size();
         i++) {
        init.emplace_back(measuredSketches_[i]);
    }

    std::vector<Ref<Sketch>> v1 = std::move(init);
    std::vector<Ref<Sketch>> v2;
    v2.reserve(v1.size());
    typedef std::pair<Ref<Sketch>, double> SketchPred;
    std::vector<SketchPred> heap;
    std::set<size_t> heapHashes(measuredHashes_);
    auto cmp = [](const SketchPred &a, const SketchPred &b) {
        return a.second > b.second;
    };
    std::vector<std::default_random_engine> gens;
    for (size_t i = 0; i < EVOLUTIONARY_SEARCH_POPULATION; i++) {
        gens.emplace_back((i + i) * randGen_());
    }
    for (int i = 0; i <= EVOLUTIONARY_SEARCH_ITERS; i++) {
        if (verbose_ >= 1) {
            logger() << "search round " << i << std::endl;
        }
        auto pred = getPrediction(v1);
        auto probSum = getProbSum(pred);
        for (size_t j = 0; j < v1.size(); j++) {
            size_t hash = v1[j]->hash();
            auto flops = pred[j];
            if (!heapHashes.count(hash)) {
                if (heap.size() < outSize) {
                    heapHashes.insert(hash);
                    heap.emplace_back(v1[j], flops);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (flops > heap[0].second) {
                    heapHashes.erase(heap[0].first->hash());
                    heapHashes.insert(hash);
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = std::make_pair(v1[j], flops);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }
        if (i == EVOLUTIONARY_SEARCH_ITERS) {
            break;
        }

        while (v2.size() < EVOLUTIONARY_SEARCH_POPULATION) {
            std::vector<Ref<Sketch>> now(EVOLUTIONARY_SEARCH_POPULATION);
            if (verbose_ >= 1) {
                logger() << "evo " << v2.size() << std::endl;
            }
#pragma omp parallel for
            for (size_t j = 0; j < EVOLUTIONARY_SEARCH_POPULATION; j++) {
                double r = randomDouble(gens[j]);
                if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB) {
                    int a = randWithProb(probSum, gens[j]);
                    auto nw = v1[a]->genMutation(gens[j]);
                    if (nw.first) {
                        try {
                            nw.second.genCode(target_);
                            now[j] = Ref<Sketch>::make(std::move(nw.second));
                        } catch (const std::exception &e) {
                        }
                    }
                } else if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB +
                                   EVOLUTIONARY_SEARCH_CROSSOVER_PROB) {
                    int a = randWithProb(probSum, gens[j]);
                    int b = randWithProb(probSum, gens[j]);
                    while (b == a)
                        b = randWithProb(probSum, gens[j]);
                    auto nw = v1[a].get()->genCrossover(*v1[b], gens[j]);
                    if (nw.first) {
                        try {
                            nw.second.genCode(target_);
                            now[j] = Ref<Sketch>::make(std::move(nw.second));
                        } catch (const std::exception &e) {
                        }
                    }
                } else {
                    now[j] = v1[randomInt(v1.size() - 1, gens[j])];
                }
            }
            for (size_t j = 0; j < EVOLUTIONARY_SEARCH_POPULATION; j++) {
                if (now[j].isValid() && !now[j]->code().empty()) {
                    v2.push_back(now[j]);
                }
            }
        }

        v1.swap(v2);
        v2.clear();
    }
    std::sort(heap.begin(), heap.end(), cmp);
    std::vector<Ref<Sketch>> ret;
    for (auto &i : heap) {
        ret.push_back(std::move(i.first));
    }
    return ret;
}

std::vector<double>
AutoSchedule::getPrediction(std::vector<Ref<Sketch>> &sketches_in) {
    std::vector<int> index;
    std::vector<double> ret(sketches_in.size());
    index.reserve(sketches_in.size());
    std::vector<Ref<Sketch>> sketches;
#pragma omp parallel for
    for (size_t i = 0; i < sketches_in.size(); i++) {
        sketches_in[i]->genSchedule();
    }
    for (size_t i = 0; i < sketches_in.size(); i++) {
        if (sketches_in[i]->schedule().ast().isValid()) {
            index.push_back(i);
            sketches.push_back(sketches_in[i]);
        } else {
            ret[i] = -1e30;
        }
    }
    if (verbose_ >= 1) {
        logger() << "get prediction" << std::endl;
    }
    auto featureList = genFeatures(sketches);
    if (verbose_ >= 1) {
        logger() << "got prediction" << std::endl;
    }
    auto predList = predictFunc_(featureList);
    for (size_t i = 0; i < predList.size(); i++) {
        ret[index[i]] = predList[i];
    }
    return ret;
}

void AutoSchedule::genSketches() {
    auto targets = findMultiLevelTiling(original_.ast());
    if (targets.empty()) {
        return;
    }
    Sketch initSketch(original_, targets);
    std::queue<Sketch> q;
    q.push(std::move(initSketch));
    while (!q.empty()) {
        auto nowSketch = std::move(q.front());
        q.pop();
        for (auto &[name, rule] : rules_) {
            if (auto status = rule->analyze(nowSketch);
                status != RuleStatus::Skip) {
                auto sketches = rule->genPart(nowSketch);
                for (auto &sketch : sketches) {
                    if (sketch.nowTargetNum() == -1) {
                        baseSketches_.push_back(
                            Ref<Sketch>::make(std::move(sketch)));
                    } else {
                        q.push(std::move(sketch));
                    }
                }
                if (status == RuleStatus::ApplyAndSkipRest) {
                    break;
                }
            }
        }
    }
}

Sketch AutoSchedule::getInitSketch() {
    auto targets = findMultiLevelTiling(original_.ast());
    if (!targets.size()) {
        return {};
    }
    return {original_, targets};
}

Schedule
AutoSchedule::testRound(const std::unordered_map<std::string, int> &nthSketch) {
    std::default_random_engine rng;
    auto sketch = getInitSketch();
    for (auto &[name, rule] : rules_) {
        if (auto status = rule->analyze(sketch); status != RuleStatus::Skip) {
            auto newSketches = rule->genPart(sketch);
            if (nthSketch.count(name)) {
                sketch = std::move(newSketches.at(nthSketch.at(name)));
            } else {
                ASSERT(!newSketches.empty());
                sketch = std::move(newSketches.front());
            }
        }
    }
    for (auto &&[type, part] : sketch.part(0)) {
        part->genFakeAnnotation(rng);
    }
    return sketch.genSchedule();
}

} // namespace freetensor
