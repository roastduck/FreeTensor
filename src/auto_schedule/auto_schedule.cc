#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>

namespace ir {

void AutoSchedule::setParams(
    const std::vector<Array *> &args,
    const std::unordered_map<std::string, Array *> &kws) {
    args_ = args;
    kws_ = kws;
    paramsSet_ = true;
}

std::vector<double> AutoSchedule::measure(const std::vector<Sketch> &sketches) {
    // Compile in parallel, and measure sequentially
    // TODO: Parallel among computing nodes

    size_t n = sketches.size();
    std::vector<Ref<Driver>> drivers(n);

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            auto func = lower(sketches[i].genSchedule().func(), target_);
            std::string code;
            if (target_->type() == TargetType::GPU)
                code = codeGenCUDA(func);
            else
                code = codeGenCPU(func);
            drivers[i] = Ref<Driver>::make(Driver(func, code, device_));
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR: " << e.what() << std::endl;
            exit(-1);
        }
    }

    std::vector<double> times;
    times.reserve(n);
    for (size_t i = 0; i < n; i++) {
        drivers[i]->setParams(args_, kws_);
        times.emplace_back(drivers[i]->time(5, 20));
    }
    return times;
}

std::pair<std::vector<std::vector<int>>, std::vector<double>>
AutoSchedule::init(int nCandidates) {
    nCandidates_ = nCandidates;
    if (!paramsSet_) {
        ERROR("Please set params first");
    }

    Sketch sketch(schedule_);
    MultiLevelTilingRule rule;
    int n = rule.analyze(schedule_);
    std::cout << "Found" << n << std::endl;
    sketch.addPart(rule.genPart(0));

    candidates_.reserve(nCandidates_);
    for (size_t i = 0; i < nCandidates_; i++) {
        candidates_.emplace_back(sketch.genRandAnnotation());
    }
    std::vector<double> times = measure(candidates_);
    for (size_t i = 0; i < nCandidates_; i++) {
        candidates_[i].setTime(times[i]);
    }

    std::vector<std::vector<int>> annotations;
    annotations.reserve(candidates_.size());
    for (auto &&candidate : candidates_) {
        annotations.emplace_back(candidate.getAnnotation());
    }

    std::sort(candidates_.begin(), candidates_.end());
    std::cout << "Initial: min " << candidates_[0].time() << " , max "
              << candidates_[nCandidates_ - 1].time() << std::endl;
    mn_ = candidates_[0].time();
    std::make_heap(candidates_.begin(), candidates_.begin() + nCandidates_);
    return std::make_pair(annotations, times);
}

std::vector<Sketch> AutoSchedule::getRandomSketches(size_t n) {
    std::vector<Sketch> ret;
    while (ret.size() < n) {
        int mut = random_int(1);
        if (mut) {
            auto nw = candidates_[random_int(nCandidates_ - 1)].genMutation();
            if (nw.first) {
                ret.push_back(nw.second);
            }
        } else {
            int a = random_int(nCandidates_ - 1);
            int b = random_int(nCandidates_ - 1);
            while (b == a)
                b = random_int(nCandidates_ - 1);
            auto nw = candidates_[a].genCrossover(candidates_[b]);
            if (nw.first) {
                ret.push_back(nw.second);
            }
        }
    }
    return ret;
}

std::pair<std::vector<std::vector<int>>, std::vector<double>>
AutoSchedule::testAndAdd(const std::vector<Sketch> &sketches) {
    std::vector<double> times = measure(sketches);
    std::vector<std::vector<int>> annotations;
    annotations.reserve(sketches.size());
    for (size_t i = 0, iEnd = sketches.size(); i < iEnd; i++) {
        annotations.emplace_back(sketches[i].getAnnotation());
        if (times[i] < candidates_[0].time()) {
            std::pop_heap(candidates_.begin(),
                          candidates_.begin() + nCandidates_);
            candidates_[nCandidates_ - 1] = sketches[i];
            candidates_[nCandidates_ - 1].setTime(times[i]);
            std::push_heap(candidates_.begin(),
                           candidates_.begin() + nCandidates_);
            mn_ = std::min(times[i], mn_);
        }
    }
    std::cout << "min " << mn_ << " max " << candidates_[0].time() << std::endl;
    return std::make_pair(annotations, times);
}

Schedule AutoSchedule::getBestSchedule() {
    int best = 0;
    int time = candidates_[0].time();
    for (size_t i = 0; i < candidates_.size(); i++) {
        if (candidates_[i].time() < time) {
            time = candidates_[i].time();
            best = i;
        }
    }
    return candidates_[best].genSchedule();
}

} // namespace ir
