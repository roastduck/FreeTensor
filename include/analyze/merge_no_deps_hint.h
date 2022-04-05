#ifndef MERGE_NO_DEPS_HINT
#define MERGE_NO_DEPS_HINT

#include <stmt.h>

namespace ir {

/**
 * When we merge or fuse some loops, we also merge the no_deps hints on them
 *
 * These are 3 cases:
 *
 * 1. If a variable is marked as no_deps on all the loops, keep it
 * 2. If a variable is marked as no_deps on some of the loops, and it is
 * naturally free of dependence on the other loops, keep it
 * 3. If a variable is marked as no_deps on some of the loops, but there is
 * dependence of it on another loop, drop it
 */
std::vector<std::string> mergeNoDepsHint(const Stmt &ast,
                                         const std::vector<ID> &loops);
inline std::vector<std::string>
mergeNoDepsHint(const Stmt &ast, const ID &loop1, const ID &loop2) {
    return mergeNoDepsHint(ast, {loop1, loop2});
}

} // namespace ir

#endif // MERGE_NO_DEPS_HINT
