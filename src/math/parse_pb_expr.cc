#include <antlr4-runtime.h>

#include <debug.h>
#include <math/parse_pb_expr.h>
#include <mutator.h>
#include <pb_lexer.h>
#include <pb_parser.h>

namespace freetensor {

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

} // Anonymous namespace

PBFuncAST parsePBFunc(const std::string &str) {
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
        throw ParserError((std::string) "Parser error: " + e.what() +
                          "\n during parsing \"" + str + "\"");
    }
}

SimplePBFuncAST parseSimplePBFunc(const std::string &str) {
    auto ret = parsePBFunc(str);
    if (ret.size() != 1) {
        throw ParserError(str + " is not a simple PBFunc");
    }
    return ret.front();
}

} // namespace freetensor
