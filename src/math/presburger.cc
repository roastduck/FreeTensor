#include <container_utils.h>
#include <math/presburger.h>

namespace freetensor {

PBBuildExpr PBBuilder::newVar(const std::string &name = "") {
    if (name.empty())
        return PBBuildExpr("anon" + toString(anonVarNum++));
    else {
        ASSERT(!namedVars.contains(name));
        namedVars.insert(name);
        return PBBuildExpr(name);
    }
}

std::vector<PBBuildExpr> PBBuilder::newVars(int n,
                                            const std::string &prefix = "") {
    std::vector<PBBuildExpr> ret;
    ret.reserve(n);
    for (int i = 0; i < n; i++)
        ret.emplace_back(prefix.empty() ? prefix : prefix + toString(i));
}

std::string PBBuilder::getConstraintsStr() const {
    return join(constraints, " and ");
}

void PBBuilder::addConstraint(const PBBuildExpr &constraint) {
    constraints.emplace_back(constraint);
}

void PBBuilder::addConstraint(PBBuildExpr &&constraint) {
    constraints.emplace_back(std::move(constraint));
}

void PBMapBuilder::addInput(const PBBuildExpr &expr) {
    inputs.emplace_back(expr);
}
PBBuildExpr PBMapBuilder::newInput(const std::string &name) {
    auto var = PBBuilder::newVar(name);
    addInput(var);
    return var;
}
std::vector<PBBuildExpr>
PBMapBuilder::newInputs(int n, const std::string &prefix = "") {
    for (const auto &var : PBBuilder::newVars(n, prefix))
        addInput(var);
}

void PBMapBuilder::addOutput(const PBBuildExpr &expr) {
    outputs.emplace_back(expr);
}
PBBuildExpr PBMapBuilder::newOutput(const std::string &name) {
    auto var = PBBuilder::newVar(name);
    addOutput(var);
    return var;
}
std::vector<PBBuildExpr>
PBMapBuilder::newOutputs(int n, const std::string &prefix = "") {
    for (const auto &var : PBBuilder::newVars(n, prefix))
        addOutput(var);
}

PBMap PBMapBuilder::build(const PBCtx &ctx) const {
    return {ctx, "{ [" + join(inputs, ", ") + "] -> [" + join(outputs, ", ") +
                     "]: " + getConstraintsStr() + " }"};
}

void PBSetBuilder::addVar(const PBBuildExpr &expr) { vars.emplace_back(expr); }
PBBuildExpr PBSetBuilder::newVar(const std::string &name) {
    auto var = PBBuilder::newVar(name);
    addVar(var);
    return var;
}
std::vector<PBBuildExpr> PBSetBuilder::newVars(int n,
                                               const std::string &prefix = "") {
    for (const auto &var : PBBuilder::newVars(n, prefix))
        addVar(var);
}

PBSet PBSetBuilder::build(const PBCtx &ctx) const {
    return {ctx, "{ [" + join(vars, ", ") + "]: " + getConstraintsStr() + " }"};
}

} // namespace freetensor
