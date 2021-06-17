//
// Created by hitonami on 2021/6/16.
//
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <driver.h>
#include <lower.h>
#include <codegen/code_gen_cuda.h>
#include <codegen/code_gen_cpu.h>
#include <auto_schedule/utils.h>

namespace ir {

void AutoSchedule::set_params(const std::vector<Array *> &args,
                          const std::unordered_map<std::string, Array *> &kws) {
    args_ = args;
    kws_ = kws;
    params_set = true;
}

Schedule AutoSchedule::run(unsigned int round, unsigned int save,
                       unsigned int extend) {
    if (!params_set) {
        std::cout << "Please set params first." << std::endl;
        assert(false);
    }
    double init_time = measure(schedule_);
    std::cout << "Initial time: " << init_time << std::endl;
    extend += save;
    Sketch sketch(schedule_);
    MultiLevelTilingRule rule;
    int n = rule.analyze(schedule_);
    std::cout << "Found" << n << std::endl;
    sketch.add_part(rule.gen_part(0));
    candidates.reserve(extend);
    while (candidates.size() < extend) {
        Sketch nw = sketch.gen_rand_annotation();
        measure(nw);
        candidates.push_back(nw);
    }
    std::sort(candidates.begin(), candidates.end());
    std::cout << "Initial: min " << candidates[0].time << " , max " << candidates[save - 1].time << std::endl;
    double mn = candidates[0].time;
    std::make_heap(candidates.begin(), candidates.begin() + save);
    for (unsigned int i = 0; i < round; i++) {
        std::cout << "Round " << i << " ";
        int mut = random_int(1);
        if (mut) {
            auto nw =
                candidates[random_int(save - 1)].gen_mutation();
            if (nw.first) {
                auto sk = nw.second;
                double time = measure(sk);
                if (time < candidates[0].time) {
                    std::pop_heap(candidates.begin(), candidates.begin() + save);
                    candidates[save - 1] = sk;
                    std::push_heap(candidates.begin(), candidates.begin() + save);
                    mn = std::min(mn, time);
                    std::cout << "New one by mutation, min " << mn << " max " << candidates[0].time << std::endl;
                }
            }
        } else {
            int a = random_int(save - 1);
            int b = random_int(save - 1);
            while (b == a)
                b = random_int(save - 1);
            auto nw = candidates[a].gen_crossover(candidates[b]);
            if (nw.first) {
                auto sk = nw.second;
                double time = measure(sk);
                if (time < candidates[0].time) {
                    std::pop_heap(candidates.begin(), candidates.begin() + save);
                    candidates[save - 1] = sk;
                    std::push_heap(candidates.begin(), candidates.begin() + save);
                    mn = std::min(mn, time);
                    std::cout << "New one by crossover, min " << mn << " max " << candidates[0].time << std::endl;
                }
            }
        }
    }
    std::sort(candidates.begin(), candidates.begin() + save);
    return candidates[0].gen_schedule();
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

double AutoSchedule::measure(Sketch &sketch) {
    sketch.time = measure(sketch.gen_schedule());
    std::cout << "\tconsumes "<< sketch.time << std::endl;
    return sketch.time;
}

} // namespace ir