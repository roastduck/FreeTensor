#ifndef CODE_GEN_H
#define CODE_GEN_H

#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <visitor.h>

namespace ir {

class CodeGen : public Visitor {
  public:
    struct Stream {
        std::string name_;
        std::ostringstream os_;
        int nIndent_ = 0;
        std::unordered_map<std::string, Ref<Buffer>> uses_;
        std::unordered_map<std::string, int> threadDim_;
    };

  protected:
    std::vector<Stream> streamStack_, poppedStream_;

    // var name -> (stream name, buffer)
    std::unordered_map<std::string, std::pair<std::string, Ref<Buffer>>> vars_;

    void makeIndent();

    template <class T> void printList(T &&list) {
        for (size_t i = 0, iEnd = list.size(); i < iEnd; i++) {
            (*this)(list[i]);
            os() << (i < iEnd - 1 ? ", " : "");
        }
    }

    CodeGen();

    void markDef(const std::string &name, const Ref<Buffer> &buffer);
    void markUse(const std::string &name);

    void pushStream(const std::string &name);
    void popStream();

    std::ostringstream &os();
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

} // namespace ir

#endif // CODE_GEN_H
