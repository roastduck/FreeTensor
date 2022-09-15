#include <container_utils.h>
#include <math/presburger.h>

namespace freetensor {

std::ostream &operator<<(std::ostream &os, const PBBuildExpr &e) {
    os << e.expr_;
    return os;
}

PBBuildExpr PBBuilder::newVar(const std::string &name) {
    if (name.empty())
        return PBBuildExpr("anon" + toString(anonVarNum++));
    else {
        ASSERT(!namedVars.contains(name));
        namedVars.insert(name);
        return PBBuildExpr(name);
    }
}

std::vector<PBBuildExpr> PBBuilder::newVars(int n, const std::string &prefix) {
    std::vector<PBBuildExpr> ret;
    ret.reserve(n);
    for (int i = 0; i < n; i++)
        ret.push_back(
            PBBuildExpr(prefix.empty() ? prefix : prefix + toString(i)));
    return ret;
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
    inputs_.emplace_back(expr);
}
PBBuildExpr PBMapBuilder::newInput(const std::string &name) {
    auto var = PBBuilder::newVar(name);
    addInput(var);
    return var;
}
std::vector<PBBuildExpr> PBMapBuilder::newInputs(int n,
                                                 const std::string &prefix) {
    auto ret = PBBuilder::newVars(n, prefix);
    for (const auto &var : ret)
        addInput(var);
    return ret;
}

void PBMapBuilder::addOutput(const PBBuildExpr &expr) {
    outputs_.emplace_back(expr);
}
PBBuildExpr PBMapBuilder::newOutput(const std::string &name) {
    auto var = PBBuilder::newVar(name);
    addOutput(var);
    return var;
}
std::vector<PBBuildExpr> PBMapBuilder::newOutputs(int n,
                                                  const std::string &prefix) {
    auto ret = PBBuilder::newVars(n, prefix);
    for (const auto &var : ret)
        addOutput(var);
    return ret;
}

PBMap PBMapBuilder::build(const PBCtx &ctx) const {
    return {ctx, "{ [" + join(inputs_, ", ") + "] -> [" + join(outputs_, ", ") +
                     "]: " + getConstraintsStr() + " }"};
}

void PBSetBuilder::addVar(const PBBuildExpr &expr) { vars_.emplace_back(expr); }
PBBuildExpr PBSetBuilder::newVar(const std::string &name) {
    auto var = PBBuilder::newVar(name);
    addVar(var);
    return var;
}
std::vector<PBBuildExpr> PBSetBuilder::newVars(int n,
                                               const std::string &prefix) {
    auto ret = PBBuilder::newVars(n, prefix);
    for (const auto &var : ret)
        addVar(var);
    return ret;
}

PBSet PBSetBuilder::build(const PBCtx &ctx) const {
    return {ctx, "{ [" + join(vars_, ", ") + "]: " + getConstraintsStr() + " }"};
}

} // namespace freetensor
