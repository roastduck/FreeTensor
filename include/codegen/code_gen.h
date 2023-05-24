#ifndef FREE_TENSOR_CODE_GEN_H
#define FREE_TENSOR_CODE_GEN_H

#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <analyze/symbol_table.h>
#include <codegen/native_code.h>
#include <container_utils.h>
#include <visitor.h>

namespace freetensor {

struct CodeGenStream {
    std::string name_;
    std::ostringstream os_;
    int nIndent_ = 0;
    std::unordered_map<std::string, VarDef> useDefs_;
    std::unordered_set<std::string> useIters_;

    CodeGenStream();
};

template <class Stream> class CodeGen : public SymbolTable<Visitor> {
  protected:
    bool compact_;
    int indentSize_;
    std::vector<Stream> streamStack_, poppedStream_;

    std::unordered_map<std::string, std::string>
        var2Stream_; // var name -> stream name

    void makeIndent();

    template <class T> void printList(T &&list) {
        for (auto &&[i, item] : views::enumerate(list)) {
            if (i > 0) {
                os() << (compact_ ? "," : ", ");
            }
            (*this)(item);
        }
    }

    CodeGen(bool compact = false, int indentSize = 2);

    void markDef(const VarDef &op);
    void markUse(const std::string &name);
    void markUndef(const VarDef &op);

    void markDefIter(const For &op);
    void markUseIter(const std::string &name);
    void markUndefIter(const For &op);

    void pushStream(const std::string &name);
    void popStream();

    std::ostream &os();
    int &nIndent();

  public:
    void beginBlock();
    void endBlock();

    /**
     * Dump all streams to a string
     *
     * @param action : callback(stream). Do more modification to a stream.
     * Function prelude and finale can be added here
     */
    std::string
    toString(const std::function<std::string(const Stream &)> &action);
};

/**
 * Generate native code
 *
 * @param func : The AST to be lowered. It must includes function signature to
 * determine parameters and return values
 * @param target : The target architecture
 */
NativeCode codeGen(const Func &func, const Ref<Target> &target);

} // namespace freetensor

#endif // FREE_TENSOR_CODE_GEN_H
