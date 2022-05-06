#include <cmath>

#include <analyze/find_elementwise.h>
#include <analyze/fixed_length_feature.h>
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/cache_write.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling_with_fusion.h>
#include <auto_schedule/rules/skip.h>
#include <auto_schedule/rules/thread_bind.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>
#include <pybind11/numpy.h>
#include <queue>

namespace freetensor {

AutoSchedule::AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                           const Ref<Device> &device, int measuredSize,
                           py::function predictFunc, py::function updateFunc)
    : original_(schedule.clone()), target_(target), device_(device),
      measuredSize_(measuredSize), paramsSet_(false), mn_(INFINITY),
      predictFunc_(std::move(predictFunc)), updateFunc_(std::move(updateFunc)) {
    if (target->type() == TargetType::CPU) {
        rules_.push_back(new CacheWriteRule(target->type()));
        rules_.push_back(new MultiLevelTilingWithFusionRule(target->type()));
        rules_.push_back(new MultiLevelTilingRule(target->type()));
    } else {
        rules_.push_back(new CacheWriteRule(target->type()));
        rules_.push_back(new MultiLevelTilingWithFusionRule(target->type()));
        rules_.push_back(new ThreadBindRule());
    }
    rules_.push_back(new SkipRule());
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
    std::cout << "codegen" << std::endl;
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            sketches[i]->genCode(target_);
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
        std::cout << "measure " << i << std::endl;
        ASSERT(paramsSet_);
        try {
            if (!drivers[i].isValid()) {
                times.emplace_back(1e30);
                continue;
            }
            drivers[i]->setParams(args_, kws_);
            double t = drivers[i]->time(5, 20);
            times.emplace_back(t);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR measure: " << e.what() << std::endl;
            times.emplace_back(1e30);
        }
    }
    return times;
}

void AutoSchedule::searchOneRound(size_t n) {
    bool firstTime = false;
    if (baseSketches_.empty()) {
        genSketches();
        firstTime = true;
    }
    std::cout << "get init population" << std::endl;
    if (!firstTime) {
        std::vector<Ref<Sketch>> init = getInitPopulation(n);
        std::cout << "evolutionary search" << std::endl;
        std::vector<Ref<Sketch>> best = evolutionarySearch(init, n);
        testAndAdd(best);
    }
    std::vector<Ref<Sketch>> rand = getRandPopulation(firstTime ? n : n * 0.2);
    testAndAdd(rand);
    auto bs = getBestSchedule();
    auto logs = bs.logs();
    for (auto log : logs) {
        std::cout << log << std::endl;
    }
    std::cout << "now best: " << toString(bs.ast()) << std::endl;
}

py::list AutoSchedule::genFeatures(std::vector<Ref<Sketch>> &sketches) {
    size_t n = sketches.size();
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            sketches[i]->genFeature();
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR feature: " << e.what() << std::endl;
            exit(-1);
        }
    }
    py::list featureList;
    for (auto &i : sketches) {
        py::list feature;
        for (auto j : i->feature()) {
            feature.append(py::float_(j));
        }
        featureList.append(feature);
    }
    return featureList;
}

std::vector<double>
AutoSchedule::testAndAdd(std::vector<Ref<Sketch>> &sketches_in) {
    std::cout << "schedule" << std::endl;
    std::vector<Ref<Sketch>> sketches;
    for (size_t i = 0; i < sketches_in.size(); i++) {
        if (!sketches_in[i]->genCode(target_).empty()) {
            sketches.push_back(sketches_in[i]);
        }
    }
    std::cout << "feature" << std::endl;
    auto features = genFeatures(sketches);
    size_t n = sketches.size();
    ASSERT(sketches.size() == n);
    std::vector<double> times = measure(sketches);
    py::list timesList;
    py::list featuresList;
    for (size_t i = 0; i < times.size(); i++) {
        if (times[i] > 1e20) {
            continue;
        }
        timesList.append(py::float_(times[i]));
        featuresList.append(features[i]);
    }
    updateFunc_(featuresList, timesList);
    auto cmp = [](const Ref<Sketch> &a, const Ref<Sketch> &b) {
        return *a < *b;
    };
    std::make_heap(measuredSketches_.begin(), measuredSketches_.end(), cmp);
    for (size_t i = 0; i < n; i++) {
        if (times[i] > 1e20) {
            continue;
        }
        if (measuredSketches_.size() < measuredSize_) {
            measuredSketches_.emplace_back(sketches[i]);
            measuredSketches_.back()->setTime(times[i]);
            std::push_heap(measuredSketches_.begin(), measuredSketches_.end(),
                           cmp);
        } else if (times[i] < measuredSketches_[0]->time()) {
            std::pop_heap(measuredSketches_.begin(), measuredSketches_.end(),
                          cmp);
            measuredSketches_.back() = sketches[i];
            measuredSketches_.back()->setTime(times[i]);
            std::push_heap(measuredSketches_.begin(), measuredSketches_.end(),
                           cmp);
        }
        measuredHashes_.insert(sketches[i]->hash());
        mn_ = std::min(times[i], mn_);
    }

    std::cout << "min " << mn_ << " max " << measuredSketches_[0]->time()
              << std::endl;
    std::sort(measuredSketches_.begin(), measuredSketches_.end(), cmp);
    return times;
}

Schedule AutoSchedule::getBestSchedule() {
    int best = 0;
    int time = measuredSketches_[0]->time();
    for (size_t i = 0; i < measuredSketches_.size(); i++) {
        if (measuredSketches_[i]->time() < time) {
            time = measuredSketches_[i]->time();
            best = i;
        }
    }
    return measuredSketches_[best]->genSchedule();
}

std::vector<Ref<Sketch>> AutoSchedule::getRandPopulation(size_t nRand) {
    std::vector<Ref<Sketch>> ret;
    std::set<size_t> used(measuredHashes_);
    std::vector<std::default_random_engine> gens;
    for (size_t i = 0; i < nRand; i++) {
        gens.push_back(std::default_random_engine((i + i) * randGen_()));
    }
    int iter = 0;
    while (ret.size() < nRand) {
        std::vector<Ref<Sketch>> now(nRand);
        size_t nThisTurn = std::min(nRand, size_t((nRand - ret.size()) * 1.5));
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
        for (size_t i = 0; i < nThisTurn; i++) {
            if (!now[i].isValid() || now[i]->code().empty()) {
                continue;
            }
            try {
                auto driver = Ref<Driver>::make(now[i]->lowered(),
                                                now[i]->code(), device_);
                driver->setParams(args_, kws_);
                driver->time(1, 0);
            } catch (const std::exception &e) {
                continue;
            }
            size_t h = now[i]->hash();
            if (!used.count(h)) {
                used.insert(h);
                ret.push_back(now[i]);
            }
            if (ret.size() >= nRand) {
                break;
            }
        }
        if (++iter > 10) {
            break;
        }
    }
    return ret;
}

std::vector<Ref<Sketch>> AutoSchedule::getInitPopulation(size_t n) {
    std::vector<Ref<Sketch>> ret = getRandPopulation(n * INIT_RAND_RATIO);
    size_t nMeasured = std::min(n - ret.size(), measuredSketches_.size());
    for (size_t i = 0; i < nMeasured; i++) {
        ret.push_back(measuredSketches_[i]);
    }
    return ret;
}

std::vector<Ref<Sketch>>
AutoSchedule::evolutionarySearch(std::vector<Ref<Sketch>> init,
                                 size_t outSize) {
    std::vector<Ref<Sketch>> v1 = std::move(init);
    std::vector<Ref<Sketch>> v2;
    v2.reserve(v1.size());
    typedef std::pair<Ref<Sketch>, double> SketchPred;
    std::vector<SketchPred> heap;
    std::set<size_t> heapHashes(measuredHashes_);
    auto cmp = [](const SketchPred &a, const SketchPred &b) {
        return a.second < b.second;
    };

    for (int i = 0; i < EVOLUTIONARY_SEARCH_ITERS; i++) {
        std::cout << "search round " << i << std::endl;
        auto pred = getPrediction(v1);
        auto probSum = getProbSum(pred);
        for (size_t j = 0; j < v1.size(); j++) {
            size_t hash = v1[j]->hash();
            auto time = pred[j];
            if (time > 1e20) {
                continue;
            }
            if (!heapHashes.count(hash)) {
                if (heap.size() < outSize) {
                    heapHashes.insert(hash);
                    heap.emplace_back(v1[j], time);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (time < heap[0].second) {
                    heapHashes.erase(heap[0].first->hash());
                    heapHashes.insert(hash);
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = std::make_pair(v1[j], time);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }

        while (v2.size() < EVOLUTIONARY_SEARCH_POPULATION) {
            double r = randomDouble(randGen_);
            if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB) {
                int a = randWithProb(probSum, randGen_);
                auto nw = v1[a]->genMutation(randGen_);
                if (nw.first) {
                    v2.push_back(Ref<Sketch>::make(std::move(nw.second)));
                }
            } else if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB +
                               EVOLUTIONARY_SEARCH_CROSSOVER_PROB) {
                int a = randWithProb(probSum, randGen_);
                int b = randWithProb(probSum, randGen_);
                while (b == a)
                    b = randWithProb(probSum, randGen_);
                auto nw = v1[a].get()->genCrossover(*v1[b], randGen_);
                if (nw.first) {
                    v2.push_back(Ref<Sketch>::make(std::move(nw.second)));
                }
            } else {
                v2.push_back(v1[randomInt(v1.size() - 1, randGen_)]);
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
            ret[i] = 1e30;
        }
    }
    std::cout << "get prediction" << std::endl;
    auto featureList = genFeatures(sketches);
    std::cout << "got prediction" << std::endl;
    py::list predList = predictFunc_(featureList);
    for (size_t i = 0; i < predList.size(); i++) {
        ret[index[i]] = predList[i].cast<double>();
    }
    return ret;
}
void AutoSchedule::genSketches() {
    auto targets = findMultiLevelTiling(original_.ast());
    if (!targets.size()) {
        return;
    }
    Sketch initSketch(original_, targets);
    std::queue<Sketch> q;
    q.push(std::move(initSketch));
    while (!q.empty()) {
        auto nowSketch = std::move(q.front());
        q.pop();
        for (auto &rule : rules_) {
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

Stmt AutoSchedule::testCacheWrite() {
    auto sketch = getInitSketch();
    CacheWriteRule rule(target_->type());
    if (rule.analyze(sketch) == RuleStatus::Skip) {
        return sketch.schedule().ast();
    }
    auto newSketch = rule.genPart(sketch).front();
    return newSketch.schedule().ast();
}

Schedule AutoSchedule::testMultiLevelTilingWithFusion(int nLevel) {
    auto sketch = getInitSketch();
    MultiLevelTilingWithFusionRule rule(target_->type());
    if (rule.analyze(sketch) == RuleStatus::Skip) {
        return sketch.schedule();
    }
    Sketch newSketch = rule.genPart(sketch)[nLevel];
    std::cout << toString(newSketch.schedule().ast()) << std::endl;
    auto part = newSketch.part(0)[SketchPartType::MultiLevelTilingWithFusion]
                    .as<MultiLevelTilingWithFusionPart>();
    part->genSampleAnnotation();
    auto schedule = newSketch.genSchedule();
    return schedule;
}

Schedule AutoSchedule::testThreadBind() {
    auto sketch = getInitSketch();
    MultiLevelTilingWithFusionRule rule(target_->type());
    if (rule.analyze(sketch) == RuleStatus::Skip) {
        return sketch.schedule();
    }
    Sketch newSketch = rule.genPart(sketch)[0];
    std::cout << toString(newSketch.schedule().ast()) << std::endl;
    auto part = newSketch.part(0)[SketchPartType::MultiLevelTilingWithFusion]
                    .as<MultiLevelTilingWithFusionPart>();
    part->genSampleAnnotation();
    newSketch.addPart(Ref<ThreadBindPart>::make());
    auto schedule = newSketch.genSchedule();
    return schedule;
}

Schedule AutoSchedule::testCacheRead() {
    auto sketch = getInitSketch();
    MultiLevelTilingWithFusionRule rule(target_->type());
    if (rule.analyze(sketch) == RuleStatus::Skip) {
        return sketch.schedule();
    }
    Sketch newSketch = rule.genPart(sketch)[0];
    std::cout << toString(newSketch.schedule().ast()) << std::endl;
    auto part = newSketch.part(0)[SketchPartType::MultiLevelTilingWithFusion]
                    .as<MultiLevelTilingWithFusionPart>();
    part->genRandAnnotation(randGen_);
    newSketch.addPart(Ref<ThreadBindPart>::make());
    auto schedule = newSketch.genSchedule();
    std::vector<Ref<Sketch>> v;
    v.push_back(Ref<Sketch>::make(newSketch));
    auto pred = getPrediction(v);
    std::cout << toString(schedule.ast()) << std::endl;
    part->printAnnotation();
    auto func = lower(schedule.func(), target_);
    std::cout << toString(func->body_) << std::endl;
    std::cout << "lower done" << std::endl;
    return {};
}

} // namespace freetensor
