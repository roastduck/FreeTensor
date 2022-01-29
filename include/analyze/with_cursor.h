#ifndef WITH_CURSOR_H
#define WITH_CURSOR_H

#include <cursor.h>

namespace ir {

/**
 * A context for Visitor or Mutator that tracks the visiting statements stack
 * with Cursor
 *
 * Inherit this class to use. E.g., inherit WithCursor<Visitor> or
 * WithCursor<Mutator>
 *
 * This class will automatically maintains the cursor if the sub-class
 * calls WithCursor::visitStmt, which is the suggested usage
 *
 * However, in some cases, this is impossible, e.g., when the sub-class needs to
 * recurse into different sub-trees manually. In these cases, the sub-class
 * should explicitly call the pushCursor / popCursor methods
 */
template <class BaseClass> class WithCursor : public BaseClass {
    Cursor cursor_;

  protected:
    void pushCursor(const Stmt &op) { cursor_.push(op); }
    void popCursor(const Stmt &) { cursor_.pop(); }

    const Cursor &cursor() const { return cursor_; }

    typename BaseClass::StmtRetType visitStmt(const Stmt &op) override {
        if constexpr (std::is_same_v<typename BaseClass::StmtRetType, void>) {
            pushCursor(op);
            BaseClass::visitStmt(op);
            popCursor(op);
        } else {
            pushCursor(op);
            auto ret = BaseClass::visitStmt(op);
            popCursor(op);
            return ret;
        }
    }
};

class GetCursorById : public WithCursor<Visitor> {
    std::string id_;
    Cursor result_;
    bool found_ = false;

  public:
    GetCursorById(const std::string &id) : id_(id) {}

    const Cursor &result() const { return result_; }
    bool found() const { return found_; }

  protected:
    void visitStmt(const Stmt &op) override;
};

inline Cursor getCursorById(const Stmt &ast, const std::string &id) {
    GetCursorById visitor(id);
    visitor(ast);
    if (!visitor.found()) {
        throw InvalidSchedule("Statement " + id + " not found");
    }
    return visitor.result();
}

class GetCursorByFilter : public WithCursor<Visitor> {
    const std::function<bool(const Cursor &)> &filter_;
    std::vector<Cursor> results_;

  public:
    GetCursorByFilter(const std::function<bool(const Cursor &)> &filter)
        : filter_(filter) {}
    const std::vector<Cursor> &results() const { return results_; }

  protected:
    void visitStmt(const Stmt &op) override;
};

inline std::vector<Cursor>
getCursorByFilter(const Stmt &ast,
                  const std::function<bool(const Cursor &)> &filter) {
    GetCursorByFilter visitor(filter);
    visitor(ast);
    return visitor.results();
}

} // namespace ir

#endif // WITH_CURSOR_H
