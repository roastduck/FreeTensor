#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <config.h>
#include <driver/device.h>
#include <except.h>
#include <func.h>
#include <lower.h>
#include <pass/cpu/lower_parallel_reduction.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/float_simplify.h>
#include <pass/gpu/lower_parallel_reduction.h>
#include <pass/gpu/lower_vector.h>
#include <pass/gpu/make_sync.h>
#include <pass/gpu/multiplex_buffers.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/gpu/normalize_var_in_kernel.h>
#include <pass/gpu/simplex_buffers.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_heap_alloc.h>
#include <pass/make_parallel_reduction.h>
#include <pass/make_reduction.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/move_out_first_or_last_iter.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_cyclic_assign.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/scalar_prop_const.h>
#include <pass/shrink_for.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/tensor_prop_const.h>
#include <pass/use_builtin_div.h>
#include <pass/z3_simplify.h>
#include <schedule.h>
#include <serialize/load_ast.h>
#include <serialize/load_driver.h>
#include <serialize/print_ast.h>
#include <serialize/print_driver.h>

namespace freetensor {

using json = nlohmann::json;

namespace {

struct CLI {
    std::string transform;
    std::optional<std::string> inputFile;
    std::optional<std::string> outputFile;
    std::optional<std::string> infoFile;
    std::optional<int> infoFd;
    bool humanReadable = false;
    bool help = false;
    std::unordered_map<std::string, std::string> opts;
};

std::string normalize(std::string s) {
    for (char &c : s) {
        if (c == '-') {
            c = '_';
        }
    }
    return s;
}

bool startsWith(const std::string &s, const std::string &prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

[[noreturn]] void usageError(const std::string &msg) {
    std::cerr << "freetensor-transform: " << msg << "\n"
              << "Try 'freetensor-transform --help' for more information.\n";
    std::exit(2);
}

void printHelp() {
    std::cout
        << "Usage:\n"
        << "  freetensor-transform <transformation-name> [options...]\n\n"
        << "Global options:\n"
        << "  -i <input-file>       Read serialized AST from file (default: "
           "stdin)\n"
        << "  -o <output-file>      Write serialized AST to file (default: "
           "stdout)\n"
        << "  --info-file <file>    Write JSON return information to file\n"
        << "  --human-readable      Write human-readable AST; colors if output "
           "is a TTY\n"
        << "  --help                Show this help\n\n"
        << "Internal option:\n"
        << "  --info-fd <fd>        Write enveloped JSON status and return "
           "information to an open fd\n\n"
        << "Names and options may use either kebab-case or snake_case.\n\n"
        << "Available transformations:\n"
        << "  Passes:\n"
        << "    simplify, z3_simplify, pb_simplify, float_simplify,\n"
        << "    flatten_stmt_seq, move_out_first_or_last_iter, "
           "scalar_prop_const,\n"
        << "    sink_var, shrink_var, shrink_for, merge_and_hoist_if,\n"
        << "    make_reduction, make_parallel_reduction, tensor_prop_const,\n"
        << "    prop_one_time_use, remove_writes, remove_cyclic_assign,\n"
        << "    remove_dead_var, make_heap_alloc, use_builtin_div,\n"
        << "    hoist_var_over_stmt_seq\n"
        << "  Schedules:\n"
        << "    split, reorder, merge, fission, fuse, swap, blend, cache,\n"
        << "    cache_reduction, set_mem_type, var_split, var_merge, "
           "var_reorder,\n"
        << "    var_unsqueeze, var_squeeze, move_to, inline, parallelize,\n"
        << "    parallelize_as, unroll, vectorize, separate_tail, as_matmul,\n"
        << "    pluto_fuse, pluto_permute\n";
}

CLI parseCLI(int argc, char **argv) {
    CLI cli;
    if (argc <= 1) {
        cli.help = true;
        return cli;
    }

    int i = 1;
    if (std::string(argv[i]) == "--help") {
        cli.help = true;
        return cli;
    }
    cli.transform = normalize(argv[i++]);

    auto needValue = [&](const std::string &arg) -> std::string {
        if (i >= argc) {
            usageError("missing value for " + arg);
        }
        return argv[i++];
    };

    while (i < argc) {
        std::string arg = argv[i++];
        if (arg == "--help") {
            cli.help = true;
        } else if (arg == "-i") {
            cli.inputFile = needValue(arg);
        } else if (arg == "-o") {
            cli.outputFile = needValue(arg);
        } else if (arg == "--human-readable" || arg == "--human_readable") {
            cli.humanReadable = true;
        } else if (arg == "--info-file" || arg == "--info_file") {
            cli.infoFile = needValue(arg);
        } else if (arg == "--info-fd" || arg == "--info_fd") {
            cli.infoFd = std::stoi(needValue(arg));
        } else if (startsWith(arg, "--")) {
            auto raw = arg.substr(2);
            auto eq = raw.find('=');
            if (eq != std::string::npos) {
                cli.opts[normalize(raw.substr(0, eq))] = raw.substr(eq + 1);
            } else {
                cli.opts[normalize(raw)] = needValue(arg);
            }
        } else {
            usageError("unexpected argument " + arg);
        }
    }
    return cli;
}

std::string readAll(std::istream &is) {
    std::ostringstream os;
    os << is.rdbuf();
    return os.str();
}

std::string readInput(const CLI &cli) {
    if (cli.inputFile) {
        std::ifstream fin(*cli.inputFile);
        if (!fin) {
            ERROR(FT_MSG << "Cannot open input file " << *cli.inputFile);
        }
        return readAll(fin);
    }
    return readAll(std::cin);
}

std::string opt(const CLI &cli, const std::string &key,
                const std::string &defaultValue = "") {
    auto it = cli.opts.find(key);
    return it == cli.opts.end() ? defaultValue : it->second;
}

bool hasOpt(const CLI &cli, const std::string &key) {
    return cli.opts.find(key) != cli.opts.end();
}

int intOpt(const CLI &cli, const std::string &key, int defaultValue = 0) {
    return hasOpt(cli, key) ? std::stoi(opt(cli, key)) : defaultValue;
}

bool boolOpt(const CLI &cli, const std::string &key,
             bool defaultValue = false) {
    if (!hasOpt(cli, key)) {
        return defaultValue;
    }
    auto v = tolower(opt(cli, key));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

ID idOpt(const CLI &cli, const std::string &key, ID defaultValue = {}) {
    if (!hasOpt(cli, key) || opt(cli, key).empty() || opt(cli, key) == "0" ||
        opt(cli, key) == "null") {
        return defaultValue;
    }
    return ID::make(std::stoull(opt(cli, key)));
}

std::vector<ID> idListOpt(const CLI &cli, const std::string &key) {
    std::vector<ID> ret;
    std::stringstream ss(opt(cli, key));
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            ret.emplace_back(ID::make(std::stoull(item)));
        }
    }
    return ret;
}

std::vector<int> intListOpt(const CLI &cli, const std::string &key) {
    std::vector<int> ret;
    std::stringstream ss(opt(cli, key));
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            ret.emplace_back(std::stoi(item));
        }
    }
    return ret;
}

json idJson(const ID &id) {
    if (id.isValid()) {
        return (uint64_t)id;
    }
    return nullptr;
}

json idMapJson(const Schedule::IDMap &map) {
    json ret = json::object();
    for (auto &&[k, v] : map) {
        ret[std::to_string((uint64_t)k)] = idJson(v);
    }
    return ret;
}

template <class F> std::pair<AST, json> runPass(const AST &ast, F f) {
    if (ast->nodeType() == ASTNodeType::Func) {
        return {f(ast.as<FuncNode>()), json::object()};
    } else if (ast->isStmt()) {
        return {f(ast.as<StmtNode>()), json::object()};
    } else {
        ERROR("Transformation expects a Func or Stmt AST");
    }
}

Ref<Target> targetOpt(const CLI &cli) {
    if (hasOpt(cli, "target_meta")) {
        return loadTarget(opt(cli, "target_meta"));
    }
    auto target = tolower(opt(cli, "target", "cpu"));
    if (target == "cpu") {
        return Ref<CPUTarget>::make();
    }
    if (target == "gpu") {
#ifdef FT_WITH_CUDA
        int device = intOpt(cli, "device", 0);
        return Ref<Device>::make(TargetType::GPU, device)->target();
#else
        ERROR("GPU target is unavailable because FT_WITH_CUDA is disabled");
#endif
    }
    ERROR("Unrecognized target " + target);
}

std::pair<AST, json> withSchedule(const AST &ast,
                                  const std::function<json(Schedule &)> &f) {
    if (ast->nodeType() == ASTNodeType::Func) {
        Schedule s(ast.as<FuncNode>());
        json info = f(s);
        return {s.func(), info};
    } else if (ast->isStmt()) {
        Schedule s(ast.as<StmtNode>());
        json info = f(s);
        return {s.ast(), info};
    } else {
        ERROR("Schedule expects a Func or Stmt AST");
    }
}

FissionSide fissionSideOpt(const CLI &cli) {
    auto s = tolower(opt(cli, "side", "before"));
    if (s == "before") {
        return FissionSide::Before;
    } else if (s == "after") {
        return FissionSide::After;
    }
    ERROR("Unrecognized fission side " + s);
}

MoveToSide moveToSideOpt(const CLI &cli) {
    auto s = tolower(opt(cli, "side", "before"));
    if (s == "before") {
        return MoveToSide::Before;
    } else if (s == "after") {
        return MoveToSide::After;
    }
    ERROR("Unrecognized move_to side " + s);
}

ReorderMode reorderModeOpt(const CLI &cli) {
    auto s = tolower(opt(cli, "mode", "perfectonly"));
    if (s == "perfectonly" || s == "perfect_only") {
        return ReorderMode::PerfectOnly;
    } else if (s == "moveoutimperfect" || s == "move_out_imperfect") {
        return ReorderMode::MoveOutImperfect;
    } else if (s == "moveinimperfect" || s == "move_in_imperfect") {
        return ReorderMode::MoveInImperfect;
    }
    ERROR("Unrecognized reorder mode " + s);
}

VarSplitMode varSplitModeOpt(const CLI &cli) {
    auto s = tolower(opt(cli, "mode", "fixedsize"));
    if (s == "fixedsize" || s == "fixed_size") {
        return VarSplitMode::FixedSize;
    } else if (s == "relaxedsize" || s == "relaxed_size") {
        return VarSplitMode::RelaxedSize;
    }
    ERROR("Unrecognized var split mode " + s);
}

AsMatMulMode asMatMulModeOpt(const CLI &cli) {
    auto s = tolower(opt(cli, "mode", "keepmemlayout"));
    if (s == "keepmemlayout" || s == "keep_mem_layout") {
        return AsMatMulMode::KeepMemLayout;
    } else if (s == "tryvarreorder" || s == "try_var_reorder") {
        return AsMatMulMode::TryVarReorder;
    } else if (s == "trytranspose" || s == "try_transpose") {
        return AsMatMulMode::TryTranspose;
    }
    ERROR("Unrecognized as_matmul mode " + s);
}

// Convert an overload set into a generic lambda so it can be passed as a value.
// Adapted from https://tartanllama.xyz/posts/passing-overload-sets/ . We do not
// preserve noexcept here because transformations report failures with
// exceptions.
#define LIFT(foo)                                                              \
    [](auto &&...xs) -> decltype(foo(std::forward<decltype(xs)>(xs)...)) {     \
        return foo(std::forward<decltype(xs)>(xs)...);                         \
    }

std::pair<AST, json> transform(const CLI &cli, const AST &ast) {
    auto name = cli.transform;

#define PASS0(cliName, cppName)                                                \
    if (name == cliName) {                                                     \
        return runPass(ast, LIFT(cppName));                                    \
    }

    PASS0("simplify", simplify)
    PASS0("z3_simplify", z3Simplify)
    PASS0("pb_simplify", pbSimplify)
    PASS0("float_simplify", floatSimplify)
    PASS0("flatten_stmt_seq", flattenStmtSeq)
    PASS0("move_out_first_or_last_iter", moveOutFirstOrLastIter)
    PASS0("scalar_prop_const", scalarPropConst)
    PASS0("shrink_var", shrinkVar)
    PASS0("merge_and_hoist_if", mergeAndHoistIf)
    PASS0("make_reduction", makeReduction)
    PASS0("remove_cyclic_assign", removeCyclicAssign)
    PASS0("remove_dead_var", removeDeadVar)
    PASS0("make_heap_alloc", makeHeapAlloc)
    PASS0("use_builtin_div", useBuiltinDiv)
    PASS0("cpu_lower_parallel_reduction", cpu::lowerParallelReduction)
#ifdef FT_WITH_CUDA
    PASS0("gpu_lower_parallel_reduction", gpu::lowerParallelReduction)
    PASS0("gpu_normalize_threads", gpu::normalizeThreads)
    PASS0("gpu_normalize_var_in_kernel", gpu::normalizeVarInKernel)
    PASS0("gpu_lower_vector", gpu::lowerVector)
#endif

#undef PASS0

#ifdef FT_WITH_CUDA
    if (name == "gpu_make_sync") {
        auto target = targetOpt(cli).as<GPUTarget>();
        return runPass(ast, [&](auto x) { return gpu::makeSync(x, target); });
    }
    if (name == "gpu_multiplex_buffers") {
        auto target = targetOpt(cli).as<GPUTarget>();
        auto def = idOpt(cli, "def_id");
        return runPass(
            ast, [&](auto x) { return gpu::multiplexBuffers(x, target, def); });
    }
    if (name == "gpu_simplex_buffers") {
        auto def = idOpt(cli, "def_id");
        return runPass(ast,
                       [&](auto x) { return gpu::simplexBuffers(x, def); });
    }
#else
    if (name.rfind("gpu_", 0) == 0) {
        ERROR(name + " is unavailable because FT_WITH_CUDA is disabled");
    }
#endif

    if (name == "sink_var") {
        return runPass(ast, LIFT(sinkVar));
    }
    if (name == "shrink_for") {
        auto sub = idOpt(cli, "sub_ast");
        auto doSimplify = boolOpt(cli, "do_simplify", true);
        auto unordered = boolOpt(cli, "unordered", false);
        return runPass(ast, [&](auto x) {
            return shrinkFor(x, sub, doSimplify, unordered);
        });
    }
    if (name == "make_parallel_reduction") {
        auto target = targetOpt(cli);
        return runPass(
            ast, [&](auto x) { return makeParallelReduction(x, target); });
    }
    if (name == "tensor_prop_const") {
        auto both = idOpt(cli, "both_in_sub_ast");
        auto either = idOpt(cli, "either_in_sub_ast");
        return runPass(
            ast, [&](auto x) { return tensorPropConst(x, both, either); });
    }
    if (name == "prop_one_time_use") {
        auto sub = idOpt(cli, "sub_ast");
        return runPass(ast, [&](auto x) { return propOneTimeUse(x, sub); });
    }
    if (name == "remove_writes") {
        auto single = idOpt(cli, "single_def_id");
        return runPass(ast, [&](auto x) { return removeWrites(x, single); });
    }
    if (name == "hoist_var_over_stmt_seq") {
        std::optional<std::vector<ID>> ids = std::nullopt;
        if (hasOpt(cli, "together_ids")) {
            ids = idListOpt(cli, "together_ids");
        }
        return runPass(ast,
                       [&](auto x) { return hoistVarOverStmtSeq(x, ids); });
    }
    if (name == "lower") {
        auto target = targetOpt(cli);
        std::unordered_set<std::string> skipPasses;
        if (hasOpt(cli, "skip_passes")) {
            std::stringstream ss(opt(cli, "skip_passes"));
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    skipPasses.insert(item);
                }
            }
        }
        auto verbose = intOpt(cli, "verbose", 0);
        return runPass(
            ast, [&](auto x) { return lower(x, target, skipPasses, verbose); });
    }

    if (name == "split") {
        return withSchedule(ast, [&](Schedule &s) {
            auto [outer, inner] =
                s.split(idOpt(cli, "id"), intOpt(cli, "factor", -1),
                        intOpt(cli, "nparts", -1), intOpt(cli, "shift", 0));
            return json{{"outer", idJson(outer)}, {"inner", idJson(inner)}};
        });
    }
    if (name == "reorder") {
        return withSchedule(ast, [&](Schedule &s) {
            s.reorder(idListOpt(cli, "order"), reorderModeOpt(cli));
            return json::object();
        });
    }
    if (name == "merge") {
        return withSchedule(ast, [&](Schedule &s) {
            auto id = s.merge(idOpt(cli, "loop1"), idOpt(cli, "loop2"));
            return json{{"id", idJson(id)}};
        });
    }
    if (name == "fission") {
        return withSchedule(ast, [&](Schedule &s) {
            auto [first, second] = s.fission(
                idOpt(cli, "loop"), fissionSideOpt(cli), idOpt(cli, "splitter"),
                boolOpt(cli, "allow_enlarge", true));
            return json{{"first", idMapJson(first)},
                        {"second", idMapJson(second)}};
        });
    }
    if (name == "fuse") {
        return withSchedule(ast, [&](Schedule &s) {
            ID id;
            if (hasOpt(cli, "loop1")) {
                id = s.fuse(idOpt(cli, "loop0"), idOpt(cli, "loop1"),
                            boolOpt(cli, "strict", false));
            } else {
                id = s.fuse(idOpt(cli, "loop0"), boolOpt(cli, "strict", false));
            }
            return json{{"id", idJson(id)}};
        });
    }
    if (name == "swap") {
        return withSchedule(ast, [&](Schedule &s) {
            s.swap(idListOpt(cli, "order"));
            return json::object();
        });
    }
    if (name == "blend") {
        return withSchedule(ast, [&](Schedule &s) {
            s.blend(idOpt(cli, "loop"));
            return json::object();
        });
    }
    if (name == "cache" || name == "cache_reduction") {
        return withSchedule(ast, [&](Schedule &s) {
            auto stmt = idOpt(cli, "stmt");
            auto var = opt(cli, "var");
            auto mtype = parseMType(opt(cli, "mtype"));
            auto [a, b, cacheVar, vardef] =
                name == "cache" ? s.cache(stmt, var, mtype)
                                : s.cacheReduction(stmt, var, mtype);
            if (name == "cache") {
                return json{{"fill", idJson(a)},
                            {"flush", idJson(b)},
                            {"var", cacheVar},
                            {"vardef", idJson(vardef)}};
            }
            return json{{"init", idJson(a)},
                        {"reduce", idJson(b)},
                        {"var", cacheVar},
                        {"vardef", idJson(vardef)}};
        });
    }
    if (name == "set_mem_type") {
        return withSchedule(ast, [&](Schedule &s) {
            if (hasOpt(cli, "reject_indirect_access")) {
                s.setMemType(idOpt(cli, "vardef"),
                             parseMType(opt(cli, "mtype")),
                             boolOpt(cli, "reject_indirect_access"));
            } else {
                s.setMemType(idOpt(cli, "vardef"),
                             parseMType(opt(cli, "mtype")));
            }
            return json::object();
        });
    }
    if (name == "var_split") {
        return withSchedule(ast, [&](Schedule &s) {
            s.varSplit(idOpt(cli, "vardef"), intOpt(cli, "dim"),
                       varSplitModeOpt(cli), intOpt(cli, "factor", -1),
                       intOpt(cli, "nparts", -1));
            return json::object();
        });
    }
    if (name == "var_merge") {
        return withSchedule(ast, [&](Schedule &s) {
            s.varMerge(idOpt(cli, "vardef"), intOpt(cli, "dim"));
            return json::object();
        });
    }
    if (name == "var_reorder") {
        return withSchedule(ast, [&](Schedule &s) {
            s.varReorder(idOpt(cli, "vardef"), intListOpt(cli, "order"));
            return json::object();
        });
    }
    if (name == "var_unsqueeze") {
        return withSchedule(ast, [&](Schedule &s) {
            s.varUnsqueeze(idOpt(cli, "vardef"), intOpt(cli, "dim"));
            return json::object();
        });
    }
    if (name == "var_squeeze") {
        return withSchedule(ast, [&](Schedule &s) {
            s.varSqueeze(idOpt(cli, "vardef"), intOpt(cli, "dim"));
            return json::object();
        });
    }
    if (name == "move_to") {
        return withSchedule(ast, [&](Schedule &s) {
            auto [moved, outer] = s.moveTo(
                idOpt(cli, "stmt"), moveToSideOpt(cli), idOpt(cli, "dst"));
            return json{{"moved", idJson(moved)}, {"outer", idJson(outer)}};
        });
    }
    if (name == "inline") {
        return withSchedule(ast, [&](Schedule &s) {
            s.inlining(idOpt(cli, "vardef"));
            return json::object();
        });
    }
    if (name == "parallelize") {
        return withSchedule(ast, [&](Schedule &s) {
            s.parallelize(idOpt(cli, "loop"),
                          parseParallelScope(opt(cli, "parallel")),
                          boolOpt(cli, "allow_reduction", true));
            return json::object();
        });
    }
    if (name == "parallelize_as") {
        return withSchedule(ast, [&](Schedule &s) {
            s.parallelizeAs(idOpt(cli, "nest"), idOpt(cli, "reference"),
                            idOpt(cli, "def_id"));
            return json::object();
        });
    }
    if (name == "unroll") {
        return withSchedule(ast, [&](Schedule &s) {
            s.unroll(idOpt(cli, "loop"), boolOpt(cli, "immediate", false));
            return json::object();
        });
    }
    if (name == "vectorize") {
        return withSchedule(ast, [&](Schedule &s) {
            s.vectorize(idOpt(cli, "loop"));
            return json::object();
        });
    }
    if (name == "separate_tail") {
        return withSchedule(ast, [&](Schedule &s) {
            s.separateTail(boolOpt(cli, "no_duplicate_var_defs", false));
            return json::object();
        });
    }
    if (name == "as_matmul") {
        return withSchedule(ast, [&](Schedule &s) {
            if (hasOpt(cli, "backend")) {
                s.asMatMul(idOpt(cli, "loop"), asMatMulModeOpt(cli),
                           targetOpt(cli),
                           parseMatMulBackend(opt(cli, "backend")));
            } else {
                s.asMatMul(idOpt(cli, "loop"), asMatMulModeOpt(cli),
                           targetOpt(cli));
            }
            return json::object();
        });
    }
    if (name == "pluto_fuse") {
        return withSchedule(ast, [&](Schedule &s) {
            auto [id, level] = s.plutoFuse(
                idOpt(cli, "loop0"), idOpt(cli, "loop1"),
                intOpt(cli, "nest_level_0", 0), intOpt(cli, "nest_level_1", 0),
                intOpt(cli, "fusable_overlap_threshold", 1),
                intOpt(cli, "fusable_nonoverlap_tolerance", 4),
                boolOpt(cli, "do_simplify", true));
            return json{{"id", idJson(id)}, {"parallel_level", level}};
        });
    }
    if (name == "pluto_permute") {
        return withSchedule(ast, [&](Schedule &s) {
            auto [id, level] =
                s.plutoPermute(idOpt(cli, "loop"), intOpt(cli, "nest_level", 0),
                               boolOpt(cli, "do_simplify", true));
            return json{{"id", idJson(id)}, {"parallel_level", level}};
        });
    }

    usageError("unknown transformation " + cli.transform);
}

void writeTextToFile(const std::string &path, const std::string &text) {
    std::ofstream out(path);
    if (!out) {
        ERROR(FT_MSG << "Cannot open output file " << path);
    }
    out << text;
}

void writeInfoFd(int fd, const std::string &text) {
    const char *p = text.data();
    size_t n = text.size();
    while (n > 0) {
        ssize_t ret = write(fd, p, n);
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw SubprocessError(FT_MSG << "Cannot write info fd " << fd);
        }
        p += ret;
        n -= ret;
    }
}

json infoEnvelope(const json &info) {
    return json{{"ok", true}, {"info", info}};
}

json exceptionEnvelope(const std::string &type, const std::string &message) {
    return json{{"ok", false},
                {"exception", json{{"type", type}, {"message", message}}}};
}

void reportException(std::optional<int> infoFd, const std::string &type,
                     const std::string &message) {
    std::cerr << message << '\n';
    if (infoFd) {
        writeInfoFd(*infoFd, exceptionEnvelope(type, message).dump() + "\n");
    }
}

} // namespace
} // namespace freetensor

int main(int argc, char **argv) {
    std::optional<int> infoFd;
    try {
        auto cli = freetensor::parseCLI(argc, argv);
        infoFd = cli.infoFd;
        if (cli.help) {
            freetensor::printHelp();
            return 0;
        }

        auto inputText = freetensor::readInput(cli);
        freetensor::AST input = freetensor::loadAST(inputText);
        auto [output, info] = freetensor::transform(cli, input);

        std::string astText;
        if (cli.humanReadable) {
            bool color = !cli.outputFile.has_value() && isatty(STDOUT_FILENO);
            astText = freetensor::toString(output, color, true, false, true,
                                           false, false, false);
        } else {
            astText = freetensor::dumpAST(output);
        }
        std::string infoText = info.dump();

        if (cli.outputFile) {
            freetensor::writeTextToFile(*cli.outputFile, astText);
        } else {
            std::cout << astText;
            if (astText.empty() || astText.back() != '\n') {
                std::cout << '\n';
            }
        }

        if (cli.infoFile) {
            freetensor::writeTextToFile(*cli.infoFile, infoText + "\n");
        } else if (cli.infoFd) {
            freetensor::writeInfoFd(
                *cli.infoFd, freetensor::infoEnvelope(info).dump() + "\n");
        } else {
            std::cout << infoText << '\n';
        }
        return 0;
    } catch (const freetensor::AssertAlwaysFalse &e) {
        freetensor::reportException(infoFd, "AssertAlwaysFalse", e.what());
    } catch (const freetensor::InvalidSchedule &e) {
        freetensor::reportException(infoFd, "InvalidSchedule", e.what());
    } catch (const freetensor::InvalidAutoGrad &e) {
        freetensor::reportException(infoFd, "InvalidAutoGrad", e.what());
    } catch (const freetensor::DriverError &e) {
        freetensor::reportException(infoFd, "DriverError", e.what());
    } catch (const freetensor::InvalidIO &e) {
        freetensor::reportException(infoFd, "InvalidIO", e.what());
    } catch (const freetensor::InvalidProgram &e) {
        freetensor::reportException(infoFd, "InvalidProgram", e.what());
    } catch (const freetensor::SymbolNotFound &e) {
        freetensor::reportException(infoFd, "SymbolNotFound", e.what());
    } catch (const freetensor::ParserError &e) {
        freetensor::reportException(infoFd, "ParserError", e.what());
    } catch (const freetensor::SubprocessError &e) {
        freetensor::reportException(infoFd, "SubprocessError", e.what());
    } catch (const freetensor::UnexpectedQueryResult &e) {
        freetensor::reportException(infoFd, "UnexpectedQueryResult", e.what());
    } catch (const freetensor::Error &e) {
        freetensor::reportException(infoFd, "Error", e.what());
    } catch (const std::exception &e) {
        freetensor::reportException(infoFd, "std::exception", e.what());
    } catch (...) {
        freetensor::reportException(infoFd, "unknown",
                                    "Unknown non-std exception");
    }
    return 1;
}
