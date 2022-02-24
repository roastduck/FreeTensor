#ifndef IR_AUTO_SCHEDULE_ANALYZER_H
#define IR_AUTO_SCHEDULE_ANALYZER_H

#include <unordered_map>
#include <stmt.h>
#include <visitor.h>
#include <analyze/symbol_table.h>

namespace ir {
typedef std::unordered_map<std::string, std::pair<Store, Load>> StoreMap;
typedef std::unordered_map<std::string, StoreMap> ReadWriteMap;

class AutoScheduleAnalyzer : public SymbolTable<Visitor> {
    ReadWriteMap reads_;
    ReadWriteMap writes_;
    Store nowStore_;
    typedef SymbolTable<Visitor> BaseClass;

  public:
    using SymbolTable<Visitor>::visit;
    void visit(const Store &op) override;
    void visit(const Load &op) override;
    StoreMap getConsumersOf(const std::string &name);
    StoreMap getProducersOf(const std::string &name);
    bool isElementWise(const Store &st, const Load &ld);
};

}  // namespace ir

#endif // IR_AUTO_SCHEDULE_ANALYZER_H
