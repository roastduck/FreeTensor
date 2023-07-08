#include <autograd/invert_from_store.h>
#include <hash.h>
#include <visitor.h>

namespace freetensor {

namespace {

class HashFinder : public Visitor {
    Expr pattern_;
    bool found_ = false;

  public:
    HashFinder(const Expr &pattern) : pattern_(pattern) {}

    bool found() const { return found_; }

  protected:
    void visitExpr(const Expr &expr) override {
        if (HashComparator{}(pattern_, expr)) {
            found_ = true;
        }
        if (found_) {
            return;
        }
        Visitor::visitExpr(expr);
    }
};

class AllReadsExcludingPattern : public Visitor {
    Expr pattern_;
    std::unordered_set<std::string> reads_;

  public:
    AllReadsExcludingPattern(const Expr &pattern) : pattern_(pattern) {}

    const auto &reads() const { return reads_; }

  protected:
    void visitExpr(const Expr &expr) override {
        if (!HashComparator{}(pattern_, expr)) {
            Visitor::visitExpr(expr);
        }
    }

    void visit(const Load &op) override {
        Visitor::visit(op);
        reads_.insert(op->var_);
    }
};

} // Anonymous namespace

bool InvertFromStore::find(const Expr &expr) const {
    HashFinder finder{yExpr_};
    finder(expr);
    return finder.found();
}

bool InvertFromStore::match(const Expr &expr) const {
    return HashComparator{}(yExpr_, expr);
}

std::unordered_set<std::string>
InvertFromStore::allReadsExcludingInversion(const Expr &expr) const {
    AllReadsExcludingPattern visitor{yExpr_};
    visitor(expr);
    return visitor.reads();
}

} // namespace freetensor
