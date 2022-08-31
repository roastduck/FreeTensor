#include <analyze/check_not_modified.h>
#include <pass/flatten_stmt_seq.h>
#include <schedule/fuse.h>
#include <schedule/pluto_fuse.h>

namespace freetensor {

std::pair<Stmt, ID> plutoFuse(const Stmt &_ast, const ID &loop0,
                              const ID &loop1, int nestLevel) {
    // flatten first so we get perfectly nested loops as much as possible
    auto ast = flattenStmtSeq(_ast);

    // check accessed vardefs: those vardefs accessed by loop1 should not have
    // their shapes modified in loop0
    CheckFuseAccessible check(loop0, loop1);
    check.check(ast);

    // count maximum count of perfectly nested loops at loop0 and loop1
    auto countPerfectNest = [](const For &loop) {
        int n = 0;
        for (auto inner = loop; inner->body_->nodeType() == ASTNodeType::For;
             inner = inner->body_.as<ForNode>())
            n++;
        return n;
    };
    int nestLevel0 = countPerfectNest(check.loop0().loop_);
    int nestLevel1 = countPerfectNest(check.loop1().loop_);

    // Process nestLevel nested loops each side; default to
    if (nestLevel == -1)
        nestLevel = std::min(nestLevel0, nestLevel1);
    else if (nestLevel0 < nestLevel)
        throw InvalidSchedule(
            "PlutoFuse: loop 0 `#" + toString(loop0) +
            "` has less than required nesting levels: " + toString(nestLevel0) +
            " existed, but " + toString(nestLevel) + " required");
    else if (nestLevel1 < nestLevel)
        throw InvalidSchedule(
            "PlutoFuse: loop 1 `#" + toString(loop1) +
            "` has less than required nesting levels: " + toString(nestLevel1) +
            " existed, but " + toString(nestLevel) + " required");
}

} // namespace freetensor
