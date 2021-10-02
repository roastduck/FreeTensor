#include <cmath>

#include <iostream>

#include <analyze/fixed_length_feature.h>
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/thread_bind.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>

namespace ir {

AutoSchedule::AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                           const Device &device, int nCandidates, int nPredict)
    : original_(schedule), current_(schedule), target_(target), device_(device),
      nCandidates_(nCandidates), nPredict_(nPredict), paramsSet_(false),
      mn_(INFINITY) {
    MultiLevelTilingRule *multiLevelTilingRule = new MultiLevelTilingRule;
    ThreadBindRule *threadBindRule = new ThreadBindRule(target_);
    rules_.push_back(std::shared_ptr<Rule>(multiLevelTilingRule));
    rules_.push_back(std::shared_ptr<Rule>(threadBindRule));
    // addParts();
    // int n = multiLevelTilingRule.analyze(original_);
    // std::cout << "Found " << n << " multi-level tiling" << std::endl;
    // int m = threadBindRule.analyze(original_);
    // std::cout << "Found " << m << " thread bind" << std::endl;
    // baseSketch_.addPart(multiLevelTilingRule.genPart(0));
    // baseSketch_.addPart(threadBindRule.genPart(0));
    // TODO: 先变换再生成sketch
}

AutoSchedule::~AutoSchedule() {
    // int rulesNum = rules_.size();
    // for (int i = 0; i < rulesNum; i++) {
    //     delete rules_[i];
    // }
}

void AutoSchedule::addParts() {
    int rulesNum = rules_.size();
    Schedule currentSchedule = original_;
    for (int i = 0; i < rulesNum; i++) {
        int n = rules_[i]->analyze(currentSchedule);
        std::cout << "Found " << n << " rule " << i << " targets." << std::endl;
        SketchPart sketchPart = rules_[i]->genPart(0);
        baseSketch_.addPart(sketchPart);
        sketchPart->genRandAnnotation();
        sketchPart->apply(currentSchedule);
    }
}

void AutoSchedule::setParams(
    const std::vector<Array *> &args,
    const std::unordered_map<std::string, Array *> &kws) {
    args_ = args;
    kws_ = kws;
    paramsSet_ = true;
}

std::vector<double>
AutoSchedule::measure(const std::vector<Schedule> &schedules) {
    // Compile in parallel, and measure sequentially
    // TODO: Parallel among computing nodes

    size_t n = schedules.size();
    std::vector<Ref<Driver>> drivers(n);

    // #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            std::cout << "Start lowering the func." << std::endl;
            auto func = lower(schedules[i].func(), target_);
            std::cout << "Finish lowering the func." << std::endl;
            std::string code;
            std::cout << "Ready to generate codes." << std::endl;
            if (target_->type() == TargetType::GPU)
                code = codeGenCUDA(func);
            else
                code = codeGenCPU(func);
            std::cout << "Successfully generate codes." << std::endl;
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
        ASSERT(paramsSet_);
        drivers[i]->setParams(args_, kws_);
        times.emplace_back(drivers[i]->time(5, 20));
    }
    return times;
}

std::vector<Sketch> AutoSchedule::getRandomSketches(size_t n) {
    std::vector<Sketch> ret;
    for (size_t i = 0; i < n; i++) {
        if (candidates_.size() < nCandidates_) {
            ret.emplace_back(baseSketch_.genRandAnnotation());
        } else {
            int mut = random_int(1);
            if (mut) {
                auto nw =
                    candidates_[random_int(nCandidates_ - 1)].genMutation();
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
    }
    return ret;
}

std::vector<Schedule>
AutoSchedule::genSchedules(const std::vector<Sketch> &sketches) {
    size_t n = sketches.size();
    std::vector<Schedule> ret(n);
    //#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            ret[i] = sketches[i].genSchedule(original_);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR: " << e.what() << std::endl;
            exit(-1);
        }
    }
    return ret;
}

std::vector<std::vector<double>>
AutoSchedule::genFeatures(const std::vector<Schedule> &schedules) {
    size_t n = schedules.size();
    std::vector<std::vector<double>> ret(n);
    //#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            ret[i] = fixedLengthFeature(schedules[i].ast());
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR: " << e.what() << std::endl;
            exit(-1);
        }
    }
    return ret;
}

std::vector<double>
AutoSchedule::testAndAdd(const std::vector<Sketch> &sketches,
                         const std::vector<Schedule> &schedules) {
    size_t n = schedules.size();
    ASSERT(sketches.size() == n);
    std::cout << "Start mesuring schedules." << std::endl;
    std::vector<double> times = measure(schedules);
    std::cout << "Successfully mesure all schedules." << std::endl;
    for (size_t i = 0; i < n; i++) {
        if (candidates_.size() < nCandidates_) {
            candidates_.emplace_back(sketches[i]);
            candidates_.back().setTime(times[i]);
            std::push_heap(candidates_.begin(), candidates_.end());
        } else if (times[i] < candidates_[0].time()) {
            std::pop_heap(candidates_.begin(), candidates_.end());
            candidates_.back() = sketches[i];
            candidates_.back().setTime(times[i]);
            std::push_heap(candidates_.begin(), candidates_.end());
        }
        mn_ = std::min(times[i], mn_);
    }
    std::cout << "min " << mn_ << " max " << candidates_[0].time() << std::endl;
    return times;
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
    return candidates_[best].genSchedule(original_);
}

int AutoSchedule::getBestTime() {
    int time = candidates_[0].time();
    for (size_t i = 0; i < candidates_.size(); i++) {
        if (candidates_[i].time() < time) {
            time = candidates_[i].time();
        }
    }
    return time;
}

int AutoSchedule::getRulesNum() { return rules_.size(); }

int AutoSchedule::ruleAnalyse(int idx) {
    return rules_[idx]->analyze(current_);
}

void AutoSchedule::ruleApply(int idx, std::vector<int> sketchPartIdx) {
    for (unsigned i = 0; i < sketchPartIdx.size(); i++) {
        SketchPart sketchPart = rules_[idx]->genPart(sketchPartIdx[i]);
        baseSketch_.addPart(sketchPart);
        sketchPart->genRandAnnotation();
        sketchPart->apply(current_);
    }
}

std::shared_ptr<Rule> AutoSchedule::getRule(int idx) { return rules_[idx]; }

void AutoSchedule::setRule(int idx, std::shared_ptr<Rule> rule) {
    rules_[idx] = rule;
}

} // namespace ir
