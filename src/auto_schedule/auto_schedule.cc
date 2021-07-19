#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>

namespace ir {

void AutoSchedule::set_params(
    const std::vector<Array *> &args,
    const std::unordered_map<std::string, Array *> &kws) {
    args_ = args;
    kws_ = kws;
    params_set = true;
}

double AutoSchedule::measure(const Schedule &schedule) {
    auto func = lower(schedule.func(), target);
    std::string code;
    if (target->type() == TargetType::GPU)
        code = codeGenCUDA(func);
    else
        code = codeGenCPU(func);
    Driver driver(func, code, device);
    driver.setParams(args_, kws_);
    return driver.time(5, 20);
}

double AutoSchedule::measure(const Sketch &sketch) {
    double time = measure(sketch.gen_schedule());
    std::cout << "\tconsumes " << time << std::endl;
    return time;
}
std::pair<std::vector<std::vector<int>>, std::vector<double>>
AutoSchedule::init(int _n_candidates) {
    n_candidates = _n_candidates;
    if (!params_set) {
        std::cout << "Please set params first." << std::endl;
        assert(false);
    }
    std::vector<std::vector<int>> annotations;
    std::vector<double> times;
    double init_time = measure(schedule_);
    std::cout << "Initial time: " << init_time << std::endl;
    Sketch sketch(schedule_);
    MultiLevelTilingRule rule;
    int n = rule.analyze(schedule_);
    std::cout << "Found" << n << std::endl;
    sketch.add_part(rule.gen_part(0));
    candidates.reserve(n);
    while (candidates.size() < n_candidates) {
        Sketch nw = sketch.gen_rand_annotation();
        double time = measure(nw);
        nw.time = time;
        candidates.push_back(nw);
        annotations.push_back(nw.get_annotation());
        times.push_back(time);
    }
    std::sort(candidates.begin(), candidates.end());
    std::cout << "Initial: min " << candidates[0].time << " , max "
              << candidates[n_candidates - 1].time << std::endl;
    mn_ = candidates[0].time;
    std::make_heap(candidates.begin(), candidates.begin() + n_candidates);
    return std::make_pair(annotations, times);
}
std::vector<Sketch> AutoSchedule::get_random_sketches(size_t n) {
    std::vector<Sketch> ret;
    while (ret.size() < n) {
        int mut = random_int(1);
        if (mut) {
            auto nw = candidates[random_int(n_candidates - 1)].gen_mutation();
            if (nw.first) {
                ret.push_back(nw.second);
            }
        } else {
            int a = random_int(n_candidates - 1);
            int b = random_int(n_candidates - 1);
            while (b == a)
                b = random_int(n_candidates - 1);
            auto nw = candidates[a].gen_crossover(candidates[b]);
            if (nw.first) {
                ret.push_back(nw.second);
            }
        }
    }
    return ret;
}
std::pair<std::vector<std::vector<int>>, std::vector<double>>
AutoSchedule::test_and_add(const std::vector<Sketch> &sketches) {
    std::vector<std::vector<int>> annotations;
    std::vector<double> times;
    for (auto &sketch : sketches) {
        double time = measure(sketch);
        annotations.push_back(sketch.get_annotation());
        times.push_back(time);
        if (time < candidates[0].time) {
            std::pop_heap(candidates.begin(),
                          candidates.begin() + n_candidates);
            candidates[n_candidates - 1] = sketch;
            candidates[n_candidates - 1].time = time;
            std::push_heap(candidates.begin(),
                           candidates.begin() + n_candidates);
            mn_ = std::min(time, mn_);
        }
    }
    std::cout << "min " << mn_ << " max " << candidates[0].time << std::endl;
    return std::make_pair(annotations, times);
}
Schedule AutoSchedule::get_best_schedule() {
    int best = 0;
    int time = candidates[0].time;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (candidates[i].time < time) {
            time = candidates[i].time;
            best = i;
        }
    }
    return candidates[best].gen_schedule();
}

} // namespace ir