#ifndef FREE_TENSOR_CODE_GEN_H
#define FREE_TENSOR_CODE_GEN_H

#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <itertools.hpp>

#include <analyze/symbol_table.h>
#include <visitor.h>

namespace freetensor {

struct CodeGenStream {
    std::string name_;
    std::ostringstream os_;
    int nIndent_ = 0;
    std::unordered_map<std::string, Ref<Buffer>> useBuffers_;
    std::unordered_set<std::string> useIters_;

    CodeGenStream();
};

template <class Stream> class CodeGen : public SymbolTable<Visitor> {
  protected:
    int indentSize_;
    std::vector<Stream> streamStack_, poppedStream_;

    std::unordered_map<std::string, std::string>
        var2Stream_; // var name -> stream name

    void makeIndent();

    template <class T> void printList(T &&list) {
        for (auto &&[i, item] : iter::enumerate(list)) {
            os() << (i > 0 ? ", " : "");
            (*this)(item);
        }
    }

    CodeGen(int indentSize = 2);

    void markDefBuffer(const VarDef &op);
    void markUseBuffer(const std::string &name);
    void markUndefBuffer(const VarDef &op);

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
std::string codeGen(const Func &func, const Ref<Target> &target);

} // namespace freetensor

#endif // FREE_TENSOR_CODE_GEN_H
