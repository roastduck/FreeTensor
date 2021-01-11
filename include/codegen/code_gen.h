#ifndef CODE_GEN_H
#define CODE_GEN_H

#include <sstream>
#include <string>

#include <visitor.h>

namespace ir {

class CodeGen : public Visitor {
  protected:
    std::ostringstream os;
    int nIndent = 0;

    void makeIndent();

    template <class T> void printList(T &&list) {
        for (size_t i = 0, iEnd = list.size(); i < iEnd; i++) {
            (*this)(list[i]);
            os << (i < iEnd - 1 ? ", " : "");
        }
    }

  public:
    void beginBlock();
    void endBlock();

    std::string toString();
};

} // namespace ir

#endif // CODE_GEN_H
