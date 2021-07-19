#include <auto_schedule/analyze/find_multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/utils.h>

namespace ir {
int MultiLevelTilingRule::analyze(Schedule &schedule) {
    FindMultiLevelTiling find;
    find(schedule.ast());
    targets = find.result();
    if (targets.empty())
        return false;
    return true;
}

SketchPart MultiLevelTilingRule::gen_part(int p) {
    return SketchPart(new MultiLevelTilingPart(targets[p]));
}

void MultiLevelTilingPart::gen_rand_annotation() {
    annotation = MultiLevelTilingAnnotation{
        random_fill_array<4>(target.i.end - target.i.begin),
        random_fill_array<4>(target.j.end - target.j.begin),
        random_fill_array<2>(target.k.end - target.k.begin)};
}

MultiLevelTilingPart::MultiLevelTilingPart(ThreeNestedFors fors) {
    target = std::move(fors);
}

template <int n>
std::array<std::pair<std::string, int>, n>
split_loop(Schedule &schedule, std::string loop, std::array<int, n> tiling) {
    std::array<std::pair<std::string, int>, n> result;
    for (int i = 0; i < n - 1; i++) {
        try {
            auto t = schedule.split(loop, tiling[i]);
            loop = t.first;
            result[n - i - 1] = std::make_pair(t.second, tiling[i]);
        } catch (const InvalidSchedule &e) {
            if (tiling[i] == 1) {
                result[n - i - 1] = std::make_pair("", 1);
            } else {
                throw e;
            }
        }
    }
    result[0] = std::make_pair(loop, tiling[n - 1]);
    return result;
}

void MultiLevelTilingPart::apply(Schedule &schedule) {
    auto i_split = split_loop<4>(schedule, target.i.id, annotation.i_tiling);
    auto j_split = split_loop<4>(schedule, target.j.id, annotation.j_tiling);
    auto k_split = split_loop<2>(schedule, target.k.id, annotation.k_tiling);

    auto tiles = {i_split[0], j_split[0], i_split[1], j_split[1], k_split[0],
                  i_split[2], j_split[2], k_split[1], i_split[3], j_split[3]};
    std::vector<std::string> labels;
    for (const auto &tile : tiles) {
        if (tile.second > 1) {
            labels.push_back(tile.first);
        }
    }
    schedule.reorder(labels);
}

SketchPart MultiLevelTilingPart::mutate() {
    MultiLevelTilingPart mut = *this;
    int mut_part = random_int(2);
    if (mut_part == 0) {
        mut.annotation.i_tiling =
            random_fill_array<4>(target.i.end - target.i.begin);
    } else if (mut_part == 1) {
        mut.annotation.j_tiling =
            random_fill_array<4>(target.j.end - target.j.begin);
    } else {
        mut.annotation.k_tiling =
            random_fill_array<2>(target.k.end - target.k.begin);
    }
    return Ref<MultiLevelTilingPart>::make(std::move(mut));
}

SketchPart MultiLevelTilingPart::crossover(const SketchPart &part) {
    if (typeid(*(part.get())) != typeid(MultiLevelTilingPart))
        return nullptr;
    auto p = part.as<MultiLevelTilingPart>();
    MultiLevelTilingPart mut = *this;
    int mut_part = random_int(2);
    if (mut_part == 0) {
        mut.annotation.i_tiling = p->annotation.i_tiling;
    } else if (mut_part == 1) {
        mut.annotation.j_tiling = p->annotation.j_tiling;
    } else {
        mut.annotation.k_tiling = p->annotation.k_tiling;
    }
    return Ref<MultiLevelTilingPart>::make(std::move(mut));
}
std::vector<int> MultiLevelTilingPart::get_annotation() const {
    std::vector<int> ret;
    ret.insert(ret.end(), annotation.i_tiling.begin(),
               annotation.i_tiling.end());
    ret.insert(ret.end(), annotation.j_tiling.begin(),
               annotation.j_tiling.end());
    ret.insert(ret.end(), annotation.k_tiling.begin(),
               annotation.k_tiling.end());
    return ret;
}

} // namespace ir
