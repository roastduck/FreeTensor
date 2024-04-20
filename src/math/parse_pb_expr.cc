#include <sstream>

#include <antlr4-runtime.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/ast_type.h>
#include <isl/union_set.h>

#include <container_utils.h>
#include <debug.h>
#include <math/parse_pb_expr.h>
#include <mutator.h>
#include <pb_lexer.h>
#include <pb_parser.h>
#include <serialize/to_string.h>

namespace freetensor {

std::ostream &operator<<(std::ostream &os, const SimplePBFuncAST &ast) {
    os << "[" << ast.args_ << "] -> [" << ast.values_ << "]";
    if (ast.cond_.isValid()) {
        os << " : " << ast.cond_;
    }
    return os;
}

namespace {

/**
 * In math/gen_pb_expr, we treat a bool variable `p` as `q > 0`, where `q` is an
 * integer variable, because ISL support integer variables only. Now we need to
 * convert it back to bool variables
 */
class RecoverBoolVars : public Mutator {
    static bool isConst0(const Expr &e) {
        return e->nodeType() == ASTNodeType::IntConst &&
               e.as<IntConstNode>()->val_ == 0;
    }

  protected:
    // q > 0 -> p
    Expr visit(const GT &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::GT);
        auto op = __op.as<GTNode>();
        if (isBool(op->lhs_->dtype())) {
            // It is not possible for ISL to find another representation of the
            // hyper-plane because we do not perform arithmetic operations on a
            // bool variable with other variables.
            ASSERT(isConst0(op->rhs_));
            return op->lhs_;
        }
        return op;
    }

    // 0 < q -> p
    Expr visit(const LT &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::LT);
        auto op = __op.as<LTNode>();
        if (isBool(op->rhs_->dtype())) {
            // It is not possible for ISL to find another representation of the
            // hyper-plane because we do not perform arithmetic operations on a
            // bool variable with other variables.
            ASSERT(isConst0(op->lhs_));
            return op->rhs_;
        }
        return op;
    }

    // q <= 0 -> !p
    Expr visit(const LE &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::LE);
        auto op = __op.as<LENode>();
        if (isBool(op->lhs_->dtype())) {
            // It is not possible for ISL to find another representation of the
            // hyper-plane because we do not perform arithmetic operations on a
            // bool variable with other variables.
            ASSERT(isConst0(op->rhs_));
            return makeLNot(op->lhs_);
        }
        return op;
    }

    // 0 >= q -> !p
    Expr visit(const GE &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::GE);
        auto op = __op.as<GENode>();
        if (isBool(op->rhs_->dtype())) {
            // It is not possible for ISL to find another representation of the
            // hyper-plane because we do not perform arithmetic operations on a
            // bool variable with other variables.
            ASSERT(isConst0(op->lhs_));
            return op->rhs_;
        }
        return op;
    }
};

PBFuncAST parsePBFuncImpl(const std::string &str) {
    try {
        antlr4::ANTLRInputStream charStream(str);
        pb_lexer lexer(&charStream);
        antlr4::CommonTokenStream tokens(&lexer);
        pb_parser parser(&tokens);
        parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());
        auto &&func = parser.func();
        auto ret = func->ast;

        RecoverBoolVars recoverer;
        for (auto &basic : ret) {
            for (auto &dim : basic.values_) {
                dim = recoverer(dim);
            }
            if (basic.cond_.isValid()) {
                basic.cond_ = recoverer(basic.cond_);
            }
        }
        return ret;
    } catch (const antlr4::ParseCancellationException &e) {
        throw ParserError(FT_MSG << "Parser error: " << e.what()
                                 << "\n during parsing \"" << str << "\"");
    }
}

} // Anonymous namespace

PBFuncAST parsePBFunc(const PBFunc::Serialized &f) {
    return parsePBFuncImpl(f.data());
}
PBFuncAST parsePBFunc(const PBSingleFunc::Serialized &f) {
    return parsePBFuncImpl(f.data());
}

namespace {

Expr isl2Expr(__isl_take isl_ast_expr *e) {
    Expr res;
    try {
        switch (isl_ast_expr_get_type(e)) {
        case isl_ast_expr_id: {
            auto id = isl_ast_expr_get_id(e);
            std::string name = isl_id_get_name(id);
            res = makeVar(name);
            isl_id_free(id);
            break;
        }
        case isl_ast_expr_int: {
            auto val = isl_ast_expr_get_val(e);
            ASSERT(isl_val_get_den_si(val) == 1);
            res = makeIntConst(isl_val_get_num_si(val));
            isl_val_free(val);
            break;
        }
        case isl_ast_expr_op: {
            auto args = views::ints(0, isl_ast_expr_op_get_n_arg(e)) |
                        views::transform([&](int i) {
                            auto result =
                                isl2Expr(isl_ast_expr_op_get_arg(e, i));
                            return result;
                        }) |
                        ranges::to_vector;
            switch (isl_ast_expr_op_get_type(e)) {
            case isl_ast_expr_op_and:
                ASSERT(args.size() == 2);
                res = makeLAnd(args[0], args[1]);
                break;
            case isl_ast_expr_op_or:
                ASSERT(args.size() == 2);
                res = makeLOr(args[0], args[1]);
                break;
            case isl_ast_expr_op_max: {
                ASSERT(!args.empty());
                Expr result = args[0];
                for (size_t i = 1; i < args.size(); ++i)
                    result = makeMax(result, args[i]);
                res = result;
            } break;
            case isl_ast_expr_op_min: {
                ASSERT(!args.empty());
                Expr result = args[0];
                for (size_t i = 1; i < args.size(); ++i)
                    result = makeMin(result, args[i]);
                res = result;
            } break;
            case isl_ast_expr_op_add:
                ASSERT(args.size() == 2);
                res = makeAdd(args[0], args[1]);
                break;
            case isl_ast_expr_op_sub:
                ASSERT(args.size() == 2);
                res = makeSub(args[0], args[1]);
                break;
            case isl_ast_expr_op_minus:
                ASSERT(args.size() == 1);
                res = makeMul(makeIntConst(-1), args[0]);
                break;
            case isl_ast_expr_op_mul:
                ASSERT(args.size() == 2);
                res = makeMul(args[0], args[1]);
                break;
            case isl_ast_expr_op_div: // Exact division. Any rounding is OK. By
                                      // defaults we use FloorDiv
            case isl_ast_expr_op_fdiv_q: // Floor division
            case isl_ast_expr_op_pdiv_q: // Floor division on non-negative
                                         // divisor
                ASSERT(args.size() == 2);
                res = makeFloorDiv(args[0], args[1]);
                break;
            case isl_ast_expr_op_pdiv_r: // Remainder on non-negative divisor.
                                         // Equivalent to Mod. We prefer Mod
                                         // over Remainder
            case isl_ast_expr_op_zdiv_r: // Divisible ? 0 : any non-zero value
                ASSERT(args.size() == 2);
                res = makeMod(args[0], args[1]);
                break;
            case isl_ast_expr_op_select:
                ASSERT(args.size() == 3);
                res = makeIfExpr(args[0], args[1], args[2]);
                break;
            case isl_ast_expr_op_eq:
                ASSERT(args.size() == 2);
                res = makeEQ(args[0], args[1]);
                break;
            case isl_ast_expr_op_le:
                ASSERT(args.size() == 2);
                res = makeLE(args[0], args[1]);
                break;
            case isl_ast_expr_op_lt:
                ASSERT(args.size() == 2);
                res = makeLT(args[0], args[1]);
                break;
            case isl_ast_expr_op_ge:
                ASSERT(args.size() == 2);
                res = makeGE(args[0], args[1]);
                break;
            case isl_ast_expr_op_gt:
                ASSERT(args.size() == 2);
                res = makeGT(args[0], args[1]);
                break;
            default:
                ASSERT(false);
            }
        } break;
        default:
            ASSERT(false);
        }
    } catch (...) {
        isl_ast_expr_free(e);
        throw;
    }
    isl_ast_expr_free(e);
    return res;
}

std::vector<std::pair<std::vector<Expr> /* values */, Expr /* cond */>>
isl2Func(__isl_take isl_ast_node *node) {
    std::vector<std::pair<std::vector<Expr>, Expr>> ret;
    try {
        if (isl_ast_node_get_type(node) == isl_ast_node_if) {
            auto cond = isl2Expr(isl_ast_node_if_get_cond(node));
            for (auto &&[thenFT, thenCond] :
                 isl2Func(isl_ast_node_if_get_then(node))) {
                ret.emplace_back(thenFT, thenCond.isValid()
                                             ? makeLAnd(cond, thenCond)
                                             : cond);
            }
            if (isl_ast_node_if_has_else(node)) {
                for (auto &&[elseFT, elseCond] :
                     isl2Func(isl_ast_node_if_get_else(node))) {
                    ret.emplace_back(elseFT,
                                     elseCond.isValid()
                                         ? makeLAnd(makeLNot(cond), elseCond)
                                         : makeLNot(cond));
                }
            }

        } else {
            // otherwise, node is a user node
            ASSERT(isl_ast_node_get_type(node) == isl_ast_node_user);
            auto expr = isl_ast_node_user_get_expr(node);
            try {
                ASSERT(isl_ast_expr_get_type(expr) == isl_ast_expr_op);
                ASSERT(isl_ast_expr_op_get_type(expr) == isl_ast_expr_op_call);
                auto nVals =
                    isl_ast_expr_op_get_n_arg(expr) -
                    1; // Arguments of the user node is values we need. The
                       // first arumgnet of the user node is its name
                auto vals =
                    views::ints(1, nVals + 1) | views::transform([&](int i) {
                        return isl2Expr(isl_ast_expr_op_get_arg(expr, i));
                    }) |
                    ranges::to_vector;

                ret = {{vals, nullptr}};
            } catch (...) {
                isl_ast_expr_free(expr);
                throw;
            }
            isl_ast_expr_free(expr);
        }
    } catch (...) {
        isl_ast_node_free(node);
        throw;
    }
    isl_ast_node_free(node);
    return ret;
}

} // Anonymous namespace

PBFuncAST parsePBFuncReconstructMinMax(const PBSet &set) {
    // This is a hack to isl's schedule. Treat the set as an iteration domain.
    // For a single-valued set, the domain will be zero or one statement,
    // implemented by a statement in multiple branches. We can recover Expr from
    // the statement and the branches' conditions.

    if (set.empty()) {
        // It will result in empty block node in isl, which we cannot parse
        return {};
    }

    ASSERT(set.isSingleValued());

    std::vector<std::string> params =
        views::ints(0, set.nParamDims()) |
        views::transform([&](int i) -> std::string {
            return isl_set_get_dim_name(set.get(), isl_dim_param, i);
        }) |
        ranges::to<std::vector>();

    isl_options_set_ast_build_detect_min_max(set.ctx()->get(), 1);

    PBFuncAST ret;
    isl_ast_build *build = isl_ast_build_alloc(set.ctx()->get());
    try {
        isl_schedule *s =
            isl_schedule_from_domain(isl_union_set_from_set(set.copy()));
        isl_ast_node *ast =
            isl_ast_build_node_from_schedule(build /* keep */, s /* take */);
        for (auto &&[vals, cond] : isl2Func(ast /* take */)) {
            ret.emplace_back(params, vals, cond);
        }
    } catch (...) {
        isl_ast_build_free(build);
        throw;
    }
    isl_ast_build_free(build);

    return ret;
}

namespace {

template <PBMapRef T> PBMap moveAllInputDimsToParam(T &&map) {
    // A name is required for the parameter, so we can't simply use
    // isl_map_move_dims. We constuct a map to apply on the set to move the
    // dimension. Example map: [i1, i2] -> {[i1, i2] -> []}. The parameters are
    // assigned with temporary names.

    int nInDims = map.nInDims();
    std::ostringstream os;
    os << "["
       << (views::ints(0, nInDims) | views::transform([](int i) {
               return "ft_unnamed_in_dim_" + std::to_string(i);
           }) |
           join(","))
       << "] -> {["
       << (views::ints(0, nInDims) | views::transform([](int i) {
               return "ft_unnamed_in_dim_" + std::to_string(i);
           }) |
           join(","))
       << "] -> []}";
    PBMap moving(map.ctx(), os.str());
    return applyDomain(std::forward<T>(map), std::move(moving));
}

} // Anonymous namespace

PBFuncAST parsePBFuncReconstructMinMax(const PBMap &map) {
    return parsePBFuncReconstructMinMax(range(moveAllInputDimsToParam(map)));
}

} // namespace freetensor
