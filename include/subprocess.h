#ifndef FREE_TENSOR_SUBPROCESS_H
#define FREE_TENSOR_SUBPROCESS_H

#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include <ast.h>
#include <except.h>
#include <func.h>
#include <id.h>
#include <stmt.h>

namespace freetensor {

struct SubprocessResult {
    AST ast_;
    std::optional<nlohmann::json> info_;
    std::string stderr_;
};

bool shouldRunInSubprocess(const std::optional<bool> &asSubprocess,
                           const std::optional<double> &timeout);

SubprocessResult
runTransformSubprocess(const std::string &name, const AST &input,
                       const std::vector<std::string> &args = {},
                       const std::optional<double> &timeout = std::nullopt);

inline std::string idArg(const ID &id) { return std::to_string((uint64_t)id); }
inline const char *boolArg(bool x) { return x ? "true" : "false"; }

inline std::vector<std::string>
subprocessArgs(std::initializer_list<std::string> xs) {
    return std::vector<std::string>(xs);
}

template <class T> T subprocessASTResult(const SubprocessResult &result) {
    if constexpr (std::is_same_v<T, Func>) {
        return result.ast_.as<FuncNode>();
    } else {
        return result.ast_.as<StmtNode>();
    }
}

template <class T>
T runPassSubprocess(const std::string &name, const T &ast,
                    const std::vector<std::string> &args,
                    const std::optional<bool> &asSubprocess,
                    const std::optional<double> &timeout) {
    if (shouldRunInSubprocess(asSubprocess, timeout)) {
        return subprocessASTResult<T>(
            runTransformSubprocess(name, ast, args, timeout));
    }
    ERROR("runPassSubprocess should only be called in subprocess mode");
}

} // namespace freetensor

#endif // FREE_TENSOR_SUBPROCESS_H
